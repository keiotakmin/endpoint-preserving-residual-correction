"""Rolling constrained least-squares controls for a shared shadow forecaster."""
import numpy as np

POLICIES = ['always','warm32','global_hard','channel_hard','global_blend','channel_blend']

class RollingControl:
    def __init__(self, channels, scope='channel', kind='blend', window=32):
        if scope not in ('global','channel') or kind not in ('hard','blend'):
            raise ValueError((scope,kind))
        if channels < 1 or window < 1: raise ValueError('positive dimensions required')
        self.scope,self.kind,self.window=scope,kind,window
        d=channels if scope=='channel' else 1
        self.history=np.zeros((window,2,d)); self.sums=np.zeros((2,d)); self.t=0
    def alpha(self):
        a,b=self.sums
        if self.kind=='hard': return (a-2*b<0).astype(float)
        return np.clip(np.divide(b,a,out=np.zeros_like(b),where=a>0),0,1)
    def update(self,correction,residual):
        a=np.mean(correction**2,axis=0); b=np.mean(correction*residual,axis=0)
        if self.scope=='global': a=np.array([a.mean()]); b=np.array([b.mean()])
        value=np.stack([a,b]); slot=self.t%self.window
        self.sums+=value-self.history[slot]; self.history[slot]=value; self.t+=1
    @property
    def state_bytes(self): return self.history.nbytes+self.sums.nbytes


def replay(base,target,predictions):
    base,target,predictions=[np.asarray(x,float) for x in (base,target,predictions)]
    assert base.shape==target.shape==predictions.shape
    n,h,c=base.shape
    controllers={p:RollingControl(c,*p.split('_')) for p in POLICIES if '_' in p}
    losses={p:np.empty((n,c)) for p in POLICIES}
    alpha={p:np.empty((n,c)) for p in POLICIES}
    static=np.mean((target-base)**2,axis=1)
    for t in range(n):
        corr=predictions[t]-base[t]
        # Alpha is determined before observing the current target.
        weights={'always':np.ones(c),'warm32':np.full(c,float(t>=32))}
        weights.update({p:g.alpha() for p,g in controllers.items()})
        for p,w in weights.items():
            alpha[p][t]=w
            losses[p][t]=np.mean((target[t]-(base[t]+corr*w))**2,axis=0)
        for g in controllers.values(): g.update(corr,target[t]-base[t])
    return static,losses,alpha,{p:g.state_bytes for p,g in controllers.items()}
