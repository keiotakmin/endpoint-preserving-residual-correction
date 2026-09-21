"""Audit and summarize Rat exclusion using archived full-period block losses only.

This is a retrospective subset analysis, not new training or independent validation.
The same script can run in the public repository with --source and --output.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def summarize(records):
    summaries=[]
    controls=['static','endpoint_only','no_residual_dct','no_endpoint','no_forecast','first','middle','mean','persistence']
    for cohort in ['main','mae','sf']:
        for subset in ['all','without_rat']:
            for policy in ['raw','global']:
                groups=defaultdict(list)
                for r in records:
                    if r['cohort']!=cohort or r['policy']!=policy:continue
                    if subset=='without_rat' and r['dataset']=='bdg2':continue
                    key=(r['id'],) if cohort=='main' else (r['dataset'],r['backbone'])
                    groups[(key,r['candidate'])].append(r)
                expected=(72 if subset=='all' else 60) if cohort=='main' else (12 if subset=='all' else 10)
                pairs=[key for key,candidate in groups if candidate=='proposed']
                assert len(pairs)==expected
                for control in controls:
                    improvements=[]
                    for pair in pairs:
                        proposed=groups[(pair,'proposed')]
                        assert len(proposed)==(1 if cohort=='main' else 3)
                        p=np.mean([[float(r[m]) for m in ['mse','mae']] for r in proposed],axis=0)
                        refrows=proposed if control=='static' else groups[(pair,control)]
                        metrics=['static_mse','static_mae'] if control=='static' else ['mse','mae']
                        ref=np.mean([[float(r[m]) for m in metrics] for r in refrows],axis=0)
                        improvements.append(100*(1-p/ref))
                    values=np.asarray(improvements)
                    summaries.append(dict(cohort=cohort,subset=subset,policy=policy,control=control,
                        pairs=expected,mse_reduction=float(values[:,0].mean()),mae_reduction=float(values[:,1].mean()),
                        mse_wins=int((values[:,0]>0).sum()),mae_wins=int((values[:,1]>0).sum())))
    return summaries


def main():
    here=Path(__file__).resolve().parent
    public=(here/'results/validation').is_dir()
    source=here/'results/validation' if public else here.parents[1]/'results/tsf_edge/endpoint_review_validation'
    output=here/'runs/rat_exclusion' if public else here.parents[1]/'results/tsf_edge/endpoint_rat_exclusion'
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=source)
    parser.add_argument('--output',type=Path,default=output)
    args=parser.parse_args()
    source=args.source/'all_results.csv'
    with source.open() as handle:
        records=[r for r in csv.DictReader(handle) if r['segment']=='full' and r['cohort'] in ['main','mae','sf']]
    assert len(records)==144*21*2
    byid=defaultdict(list)
    for r in records:byid[r['id']].append(r)
    reconstructed=[];hashes={};max_error=0.
    for key,rows in sorted(byid.items()):
        file=args.source/('blocks_'+key+'.npz')
        hashes[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
        with np.load(file,allow_pickle=False) as z:
            names=list(z['names']);static=z['static'].mean(axis=0)
            for row in rows:
                index=names.index(row['candidate']);policy=int(row['policy']=='global')
                mse,mae=z['losses'][index,:,policy].mean(axis=0)
                vals=[mse,mae,*static];keys=['mse','mae','static_mse','static_mae']
                error=max(abs(float(row[k])-v) for k,v in zip(keys,vals));max_error=max(max_error,error)
                assert error<1e-12
                reconstructed.append(dict(row,**dict(zip(keys,vals))))
    summaries=summarize(records);recomputed=summarize(reconstructed)
    for a,b in zip(summaries,recomputed):
        for k in ['mse_reduction','mae_reduction']:assert abs(a[k]-b[k])<1e-10
        for k in ['pairs','mse_wins','mae_wins']:assert a[k]==b[k]
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'summary.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    audit=dict(scope='Retrospective Rat exclusion; unchanged archived forecasts; no independent validation',
        streams=len(byid),audited_full_period_rows=len(records),summary_rows=len(summaries),
        max_block_rescore_error=max_error,all_results_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_block_sha256=hashes,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        summary_sha256=hashlib.sha256((args.output/'summary.csv').read_bytes()).hexdigest())
    (args.output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='source_block_sha256'},indent=2))
    for r in summaries:
        if r['subset']=='without_rat' and r['policy']=='global' and r['control'] in ['static','endpoint_only','no_residual_dct','first','middle','mean']:print(r)


if __name__=='__main__':main()
