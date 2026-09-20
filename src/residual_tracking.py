"""Causal residual trackers. All predictions precede updates; float64 reference code."""
import numpy as np


def basis(h, k=3):
    j = np.arange(k)
    u = np.cos(np.pi * (np.arange(h)[:, None] + .5) * j / h) * np.sqrt(2/h)
    u[:, 0] /= np.sqrt(2)
    return u


METHODS = ['candidate', 'previous', 'mean', 'bias_ewma_0.1'] + [f'ewma_{r}' for r in (.01,.1,.5)] + [f'kf_{r}' for r in (.001,.01,.1,1.)]


def evaluate(base, target, method, bounded=False, traces=False):
    """KF entries use fixed dimensionless Q/R and R=1 in coefficient space.
    EWMA is also normalized LMS on an orthonormal output basis with that step.
    Only bounded=True uses the gate, with physical bounds [0,1].
    """
    base, target = np.asarray(base, dtype=float), np.asarray(target, dtype=float)
    tmax,h,c = base.shape
    if bounded and (np.min(target)<0 or np.max(target)>1):
        raise ValueError('Bounded gate requires targets in [0,1] without target clipping')
    u=basis(h); a=np.zeros((3,c)); p=np.zeros(c); w=np.full((2,c),.5)
    raw=[]; mixed=[]; frozen=[]; gains=[]; noise=[]; process=[]; weights=[]
    for t in range(tmax):
        b=base[t]; pred=b+u@a
        if bounded:
            b=np.clip(b,0,1); pred=np.clip(pred,0,1)
        mix=w[0]*b+w[1]*pred
        lb=np.mean((target[t]-b)**2,axis=0); lp=np.mean((target[t]-pred)**2,axis=0)
        raw.append(lp.mean()); frozen.append(lb.mean()); mixed.append(np.mean((target[t]-mix)**2)); weights.append(w[1].copy())
        # Saved ORIGINAL base prediction defines the residual, independent of gate.
        e=target[t]-base[t]; z=u.T@e
        r=np.sum((e-u@z)**2,axis=0)/(h-3); q=np.zeros(c)
        if method=='candidate':
            q=np.maximum(0,np.mean((z-a)**2,axis=0)-p-r)
            s=p+q; gain=np.divide(s,s+r,out=np.zeros(c),where=(s+r)>0)
            a+=gain*(z-a); p=(1-gain)*s
        elif method=='previous':
            gain=np.ones(c); a=z.copy()
        elif method=='mean':
            gain=np.full(c,1/(t+1)); a+=gain*(z-a)
        elif method=='bias_ewma_0.1':
            gain=np.full(c,.1); a[0]+=.1*(z[0]-a[0])
        elif method.startswith('ewma_'):
            gain=np.full(c,float(method.split('_')[1])); a+=gain*(z-a)
        elif method.startswith('kf_'):
            s=p+float(method.split('_')[1]); gain=s/(s+1); a+=gain*(z-a); p=1*gain
        else: raise ValueError(method)
        gains.append(gain.copy()); noise.append(r); process.append(q)
        if bounded:
            logpost=np.log(w)-.5*np.stack([lb,lp]); logpost-=logpost.max(axis=0)
            post=np.exp(logpost); post/=post.sum(axis=0)
            alpha=1/(t+2); w=(1-alpha)*post+alpha/2
    raw=np.array(raw); mixed=np.array(mixed); frozen=np.array(frozen)
    out={}
    for name,loss in [('raw',raw)]+([('gate',mixed)] if bounded else []):
        excess=loss-frozen
        out[name]={'mse':float(loss.mean()),'benefit_pct':float(100*(1-loss.mean()/frozen.mean())) if frozen.mean()>0 else None,
                   'excess_sum':float(excess.sum()),'worst_prefix_excess':float(max(0,np.cumsum(excess).max())),
                   'worst_32block_excess':float(np.convolve(excess,np.ones(32)/32,'valid').max())}
    out['static_mse']=float(frozen.mean()); out['mean_gain']=float(np.mean(gains)); out['mean_q']=float(np.mean(process)); out['mean_R']=float(np.mean(noise))
    if traces:
        out['trace']={'raw':raw,'gate':mixed,'static':frozen,'gain':np.array(gains),'q':np.array(process),'R':np.array(noise),'weight':np.array(weights)}
    return out
