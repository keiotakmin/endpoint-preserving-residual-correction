"""Verify this published snapshot and recompute numerical summaries on a new clone."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent

def csvrows(path): return list(csv.DictReader(path.open()))
def mean(values): return float(np.mean(list(values)))

def main():
    manifest=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text())
    for name,expected in manifest.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    print('Snapshot SHA-256 checks:',len(manifest),'files OK')
    rows=csvrows(ROOT/'results/paper/tableS01_all_conditions.csv')
    main=[r for r in rows if r['group']=='real'];assert len(main)==72 and len(rows)==76
    for name,col in [('Proposed','mse'),('ELF','elf_mse')]:
        print(name,'main mean MSE reduction (%):',mean(100*(1-float(r[col])/float(r['static_mse'])) for r in main))
    print('Proposed MSE/MAE wins vs static (all 76):',*[sum(float(r[m])<float(r['static_'+m]) for r in rows) for m in ('mse','mae')])
    learned=csvrows(ROOT/'results/paper/tableS02_learned_all.csv');assert len(learned)==864
    g=defaultdict(list)
    for r in learned:g[(r['kind'],r['rank'],r['dataset'],r['backbone'],r['arm'])].append(float(r['mse']))
    for kind in ('mae','sf'):
        for width in ('2','4','64'):
            gain=[100*(1-mean(v)/mean(g[(*k[:-1],'adapter_global')])) for k,v in g.items() if k[0]==kind and k[1]==width and k[-1]=='boundary']
            print('Proposed vs learned adapter:',kind,width,mean(gain),'% MSE reduction')
    positions=json.loads((ROOT/'results/endpoint_positions.json').read_text());assert len(positions)==576
    g=defaultdict(list)
    for r in positions:
        if r['policy']=='global':
            for metric in ('mse','mae'):g[(r['kind'],r['dataset'],r['backbone'],r['arm'],metric)].append(r[metric])
    for kind in ('mae','sf'):
        for arm in ('first','middle','mean'):
            gains=[]
            for metric in ('mse','mae'):
                gains.append(mean(100*(1-mean(v)/mean(g[(k[0],k[1],k[2],arm,k[4])])) for k,v in g.items() if k[0]==kind and k[3]=='endpoint' and k[4]==metric))
            print('Endpoint vs summary:',kind,arm,'MSE/MAE reductions (%)',gains)
    validation=ROOT/'results/validation'
    rows=csvrows(validation/'all_results.csv');choices=csvrows(validation/'selection.csv')
    byid=defaultdict(list);selected=defaultdict(list)
    for r in rows:byid[r['id']].append(r)
    for r in choices:selected[r['id']].append(r)
    count=0
    for file in sorted(validation.glob('blocks_*.npz')):
        with np.load(file,allow_pickle=False) as z:
            loss,static,names=z['losses'],z['static'],list(z['names'])
            key=json.loads(str(z['meta']))['id'];count+=1
            assert len(byid[key])==210 and len(selected[key])==8
            for r in byid[key]:
                lo,hi=int(r['start']),int(r['stop']);i=names.index(r['candidate']);p=int(r['policy']=='global')
                values=np.r_[loss[i,lo:hi,p].mean(0),static[lo:hi].mean(0)]
                np.testing.assert_allclose(values,[float(r[n]) for n in ('mse','mae','static_mse','static_mae')],rtol=1e-12,atol=1e-12)
                assert int(r['state_bytes'])==z['sizes'][i,p]
            for r in selected[key]:
                lo,hi=int(r['start']),int(r['stop']);n=13 if r['pool']=='settings13' else 21
                prefix=int(np.argmin(loss[:n,:lo,1,0].mean(1)))
                oracle=int(np.argmin(loss[:n,lo:hi,1,0].mean(1)))
                assert names[prefix]==r['selected'] and names[oracle]==r['oracle']
                for name,i in [('fixed',0),('selected',prefix),('oracle',oracle)]:
                    np.testing.assert_allclose(loss[i,lo:hi,1].mean(0),[float(r[name+'_'+m]) for m in ('mse','mae')],rtol=1e-12,atol=1e-12)
    assert (count,len(rows),len(choices))==(148,31080,1184)
    print('Validated:',count,'streams,',len(rows),'result rows,',len(choices),'selection records')
    for candidate in ('proposed','endpoint_only','no_residual_dct','no_forecast','no_endpoint','persistence'):
        rs=[r for r in rows if (r['cohort'],r['segment'],r['policy'],r['candidate'])==('main','full','global',candidate)]
        print(candidate,mean(100*(1-float(r['mse'])/float(r['static_mse'])) for r in rs),'% MSE reduction; median bytes',np.median([int(r['state_bytes']) for r in rs]))
    print('All checks passed. Numerical-record reproduction is distinct from retraining base models.')

if __name__=='__main__':main()
