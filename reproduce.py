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
    methods=['Static','$\\delta$-Adapter','COSA','FAC','OMPB','ELF','Proposed']
    assert len(arms)==672 and {r['method'] for r in arms}==set(methods)
    for method,expected in [('Proposed',6352),('ELF',474048)]:
        sizes=[int(r['state_bytes']) for r in arms if r['method']==method]
        assert len(sizes)==96 and np.median(sizes)==expected
    cells=defaultdict(dict);series=defaultdict(list)
    for r in arms:
        cells[(r['dataset'],r['backbone'],r['phase'],r['seed'])][r['method']]=r
        series[(r['dataset'],r['backbone'],r['method'])].append(r)
    assert len(cells)==96 and all(len(c)==7 for c in cells.values())
    overview=csvrows(ROOT/'results/paper/table03_overview.csv');assert [r['method'] for r in overview]==methods
    for r in overview:
        q=[c[r['method']] for c in cells.values()]
        np.testing.assert_allclose([mean(float(a[m+'_reduction_pct']) for a in q) for m in ('mse','mae')]+[np.median([int(a['state_bytes']) for a in q])],
                                   [float(r[k]) for k in ('mse_reduction_pct','mae_reduction_pct','bytes_median')],rtol=0,atol=1e-9)
    means=csvrows(ROOT/'results/paper/table03.csv');assert len(means)==16
    for r in means:
        for method in methods:
            q=series[(r['Dataset'],r['Backbone'],method)];assert len(q)==6
            np.testing.assert_allclose([mean(float(a[m]) for a in q) for m in ('mse','mae')],[float(r[f'{method} {m.upper()}']) for m in ('mse','mae')],rtol=0,atol=1e-12)
    print('Seven-method summaries: 96-condition means and 16 series/backbone means verified')
    for method in methods[1:-1]:
        result=[]
        for m in ('mse','mae'):
            pairs=[(float(c['Proposed'][m]),float(c[method][m])) for c in cells.values()]
            result+=[mean(100*(1-p/b) for p,b in pairs),sum(p<b for p,b in pairs)]
        print('EPOC vs',method.replace('$\\delta$','delta'),'paired MSE reduction (%):',result[0],'lower in',result[1],'/96; MAE:',result[2],'lower in',result[3],'/96')
    settings=['proposed','k1','k2','k3','k6','k8','ridge0.1','ridge10','half32','half512','halfNone','window8','window128']
    sensitivity=csvrows(ROOT/'results/paper/tableS05_sensitivity_conditions.csv');assert len(sensitivity)==96*len(settings)
    summary=csvrows(ROOT/'results/paper/table07_setting_sensitivity.csv');assert len(summary)==len(settings)
    for name,r in zip(settings,summary):
        q=[c for c in sensitivity if c['candidate']==name];assert len(q)==96
        np.testing.assert_allclose([mean(100*(1-float(c[m])/float(c['static_'+m])) for c in q) for m in ('mse','mae')]+[np.median([int(c['state_bytes']) for c in q])],
                                   [float(r[k]) for k in ('MSE (\\%)','MAE (\\%)','Bytes')],rtol=0,atol=1e-9)
        if name=='proposed':
            for c in q:
                reference=cells[(c['dataset'],c['backbone'],c['phase'],c['seed'])]['Proposed']
                np.testing.assert_allclose([float(c[m]) for m in ('mse','mae')],[float(reference[m]) for m in ('mse','mae')],rtol=0,atol=1e-12)
    print('Setting sensitivity:',len(settings),'settings x 96 conditions verified')
    learned=csvrows(ROOT/'results/paper/tableS02_learned_all.csv');assert len(learned)==864
    training=csvrows(ROOT/'results/paper/tableS02_width64_training.csv');assert len(training)==288
    g=defaultdict(list);sizes=defaultdict(list)
    for r in learned:
        g[(r['kind'],r['rank'],r['dataset'],r['backbone'],r['arm'])].append([float(r['mse']),float(r['mae'])])
        sizes[(r['kind'],r['rank'],r['arm'])].append(int(r['state_bytes']))
    for r in training:
        if r['arm'].endswith('_base'):g[(r['arm'][:-5],'base-only',r['dataset'],r['backbone'])].append([float(r['mse']),float(r['mae'])])
    def reduction(first,reference):return 100*(1-np.mean(first,0)/np.mean(reference,0))
    joint=csvrows(ROOT/'results/paper/table04_tefl_joint.csv');assert len(joint)==6
    for r in joint:
        kind,width=r['Warmup'],r['Width']
        pairs=sorted({k[2:4] for k in g if k[:2]==(kind,width)});assert len(pairs)==12
        same=np.mean([reduction(g[(kind,width,*p,'boundary')],g[(kind,width,*p,'adapter_global')]) for p in pairs],0)
        separate=np.mean([reduction(g[(kind,width,*p,'adapter')],g[(kind,'base-only',*p)]) for p in pairs],0)
        np.testing.assert_allclose([*same,*separate,np.median(sizes[(kind,width,'boundary')]),np.median(sizes[(kind,width,'adapter_global')])],
                                   [float(r[k]) for k in ('MSE reduction vs adapter (%)','MAE reduction vs adapter (%)','Adapter vs base-only MSE reduction (%)',
                                                         'Adapter vs base-only MAE reduction (%)','Proposed median B','Adapter median B')],rtol=0,atol=1e-6)
        print('EPOC vs learned adapter:',kind,width,'MSE/MAE reductions (%)',list(same))
    positions=json.loads((ROOT/'results/endpoint_positions.json').read_text());assert len(positions)==576
    g=defaultdict(list)
    for r in positions:
        if r['policy']=='global':
            for metric in ('mse','mae'):g[(r['kind'],r['dataset'],r['backbone'],r['arm'],metric)].append(r[metric])
    selection={(r['Base / training'],r['Equal-state control']):r for r in csvrows(ROOT/'results/paper/table05_endpoint_selection.csv')}
    assert len(selection)==9
    for kind,cohort in (('mae','Joint MAE (12 pairs)'),('sf','Joint MAE+SF (12 pairs)')):
        for arm,control in (('first','First residual'),('middle','Middle residual'),('mean','Block mean')):
            gains=[]
            for metric in ('mse','mae'):
                gains.append(mean(100*(1-mean(v)/mean(g[(k[0],k[1],k[2],arm,k[4])])) for k,v in g.items() if k[0]==kind and k[3]=='endpoint' and k[4]==metric))
            np.testing.assert_allclose(gains,[float(selection[(cohort,control)][k]) for k in ('MSE gain (\\%)','MAE gain (\\%)')],rtol=0,atol=1e-9)
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
        print('72-condition retrospective cohort:',candidate,mean(100*(1-float(r['mse'])/float(r['static_mse'])) for r in rs),'% MSE reduction; median bytes',np.median([int(r['state_bytes']) for r in rs]))
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
