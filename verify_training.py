"""Compare newly generated full-period losses with the archived paper tables."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=ROOT/'runs/evaluation/metrics.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/verification.json')
    parser.add_argument('--atol',type=float,default=1e-6)
    args=parser.parse_args()
    def read(path):
        with path.open() as handle:return list(csv.DictReader(handle))
    mainrows=read(ROOT/'results/paper/tableS01_all_conditions.csv')
    learned=read(ROOT/'results/paper/tableS02_learned_all.csv')
    records=[]
    for r in read(args.input):
        if r['segment']!='full':continue
        key=(r['dataset'],r['backbone'],r['seed'])
        ref=None
        if r['group']=='base':
            match=[v for v in mainrows if (v['dataset'],v['backbone'],v['seed'])==key and v['phase']==r['phase']]
            if not match:continue
            ref=match[0]
            if r['candidate']=='proposed' and r['policy']=='global': columns=['mse','mae']
            elif r['candidate']=='elf_full' and r['policy']=='global':columns=['elf_mse','elf_mae']
            else:continue
        else:
            if not r['phase'].startswith('joint_r'):continue
            rank=r['phase'].removeprefix('joint_r')
            arm={('proposed','global'):'boundary',('adapter','raw'):'adapter',('adapter','global'):'adapter_global'}.get((r['candidate'],r['policy']))
            if arm is None:continue
            match=[v for v in learned if (v['dataset'],v['backbone'],v['seed'])==key and v['kind']==r['kind'] and v['rank']==rank and v['arm']==arm]
            if not match:continue
            ref=match[0];columns=['mse','mae']
        actual=[float(r[m]) for m in ('mse','mae')];expected=[float(ref[m]) for m in columns]
        difference=float(np.max(np.abs(np.array(actual)-expected)))
        records.append(dict(stream=r['stream'],candidate=r['candidate'],policy=r['policy'],
                            mse=actual[0],mae=actual[1],reference_mse=expected[0],reference_mae=expected[1],
                            max_absolute_difference=difference))
    if not records:raise ValueError('No comparable records found')
    report=dict(records=records,tolerance=args.atol,max_absolute_difference=max(r['max_absolute_difference'] for r in records))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(len(records),'comparisons; maximum absolute difference',report['max_absolute_difference'])
    if report['max_absolute_difference']>args.atol:
        raise SystemExit('Differences exceed tolerance; inspect verification.json and environment before interpreting new results')


if __name__=='__main__':main()
