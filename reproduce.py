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
    assert len(rows)==96 and len({(r['dataset'],r['backbone'],r['phase'],r['seed']) for r in rows})==96
    for name,col in [('EPOC','mse'),('ELF full','elf_mse')]:
        reduction=mean(100*(1-float(r[col])/float(r['static_mse'])) for r in rows)
        print(name,'96-condition mean MSE reduction (%):',reduction)
        assert abs(reduction-{'EPOC':15.40,'ELF full':19.29}[name])<0.005
    mae_reduction=mean(100*(1-float(r['mae'])/float(r['static_mae'])) for r in rows)
    assert abs(mae_reduction-9.35)<0.005
    print('EPOC 96-condition mean MAE reduction (%):',mae_reduction)
    print('EPOC MSE/MAE wins vs static:',*[sum(float(r[m])<float(r['static_'+m]) for r in rows) for m in ('mse','mae')])
    arms=csvrows(ROOT/'results/paper/table03_all_conditions.csv')
    assert len(arms)==672 and {r['method'] for r in arms}=={'Static','Proposed','ELF','COSA','FAC','OMPB','$\\delta$-Adapter'}
    for method,expected in [('Proposed',6352),('ELF',474048)]:
        sizes=[int(r['state_bytes']) for r in arms if r['method']==method]
        assert len(sizes)==96 and np.median(sizes)==expected
    assert len(csvrows(ROOT/'results/paper/table05_endpoint_selection.csv'))==9
    learned=csvrows(ROOT/'results/paper/tableS02_learned_all.csv');assert len(learned)==864
    g=defaultdict(list)
    for r in learned:g[(r['kind'],r['rank'],r['dataset'],r['backbone'],r['arm'])].append(float(r['mse']))
    for kind in ('mae','sf'):
        for width in ('2','4','64'):
            gain=[100*(1-mean(v)/mean(g[(*k[:-1],'adapter_global')])) for k,v in g.items() if k[0]==kind and k[1]==width and k[-1]=='boundary']
            print('EPOC vs learned adapter:',kind,width,mean(gain),'% MSE reduction')
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
    from report_rat_exclusion import summarize
    subset_rows=[r for r in rows if r['segment']=='full' and r['cohort'] in ('main','mae','sf')]
    actual=summarize(subset_rows)
    expected=csvrows(ROOT/'results/rat_exclusion/summary.csv')
    assert len(actual)==len(expected)==108
    for a,b in zip(actual,expected):
        for key in ('cohort','subset','policy','control'):assert a[key]==b[key]
        for key in ('pairs','mse_wins','mae_wins'):assert a[key]==int(b[key])
        for key in ('mse_reduction','mae_reduction'):
            np.testing.assert_allclose(a[key],float(b[key]),rtol=0,atol=1e-10)
    print('Rat-exclusion audit:',len(actual),'summaries verified; previously audited block losses unchanged')
    print('All checks passed. Numerical-record reproduction is distinct from retraining base models.')

if __name__=='__main__':main()
