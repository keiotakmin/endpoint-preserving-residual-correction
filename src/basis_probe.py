"""Bounded-slot exploration of residual basis components; exploratory heuristic."""
import numpy as np
from forgetting_regression import WeightedRidge
from channel_control import RollingControl
ARMS=['fixed2','fixed3','fixed6','gain_probe','energy_probe','cyclic_probe']

def columns(h,ids):
    ids=np.asarray(ids);u=np.cos(np.pi*(np.arange(h)[:,None]+.5)*ids/h)*np.sqrt(2/h)
    u[:,ids==0]/=np.sqrt(2)
    return u

class ProbeTracker:
    def __init__(self,h,c,arm):
        if arm not in ARMS:raise ValueError(arm)
        self.h,self.c,self.arm=h,c,arm;self.probe=arm.endswith('_probe')
        k=3 if self.probe else int(arm[-1]);self.ids=np.arange(k,dtype=np.int64)
        self.u=columns(h,self.ids);self.model=WeightedRidge((k,c),half_life=128)
        self.last=np.zeros((k,c));self.gain=np.zeros(k);self.energy=np.zeros(k)
        self.t=0;self.cursor=3;self.swaps=0;self.trials=0
    def issue(self,base):
        x=np.stack([np.ones_like(self.last),self.last,self.u.T@base],axis=-1)
        a=self.model.predict(x);active=2 if self.probe else len(self.ids)
        corr=self.u[:,:active]@a[:active]
        return corr,(x,a)
    def mature(self,residual,pending):
        x,a=pending;z=self.u.T@residual
        self.gain+=np.mean(2*z*a-a*a,axis=1);self.energy+=np.mean(z*z,axis=1)
        self.model.update(x,z);self.last=z;self.t+=1
        if not self.probe or self.t%32:return
        if self.arm=='cyclic_probe':worst=self.trials%2;promote=True
        else:
            score=self.gain if self.arm=='gain_probe' else self.energy
            worst=int(np.argmin(score[:2]));promote=score[2]>max(0.,score[worst])
        if promote:
            for ar in [self.ids,self.model.G,self.model.q,self.last]:
                ar[[worst,2]]=ar[[2,worst]]
            self.swaps+=1
        active=set(self.ids[:2].tolist())
        while self.cursor%6 in active:self.cursor+=1
        self.ids[2]=self.cursor%6;self.cursor+=1;self.trials+=1
        self.model.G[2]=np.eye(3);self.model.q[2]=0
        self.u=columns(self.h,self.ids)
        self.last[2]=self.u[:,2]@residual
        self.gain[:]=0;self.energy[:]=0
    @property
    def state_bytes(self):
        return sum(a.nbytes for a in [self.ids,self.model.G,self.model.q,self.last,self.gain,self.energy,self.u])

def evaluate(base,target,arm,bounded=False,trace=False):
    base,target=np.asarray(base,float),np.asarray(target,float)
    if bounded and (target.min()<0 or target.max()>1):raise ValueError('physical bound violated')
    n,h,c=base.shape;m=ProbeTracker(h,c,arm);gate=RollingControl(c,'global','blend')
    losses={p:np.empty((n,c)) for p in ['static','raw','global_blend']};ids=[];pred=[]
    for t in range(n):
        corr,token=m.issue(base[t]);b=base[t];raw=b+corr
        if bounded:b=np.clip(b,0,1);raw=np.clip(raw,0,1)
        corr=raw-b;issued=b+gate.alpha()*corr
        if trace:ids.append(m.ids.copy());pred.append(issued.copy())
        for name,p in [('static',b),('raw',raw),('global_blend',issued)]:losses[name][t]=np.mean((target[t]-p)**2,axis=0)
        gate.update(corr,target[t]-b);m.mature(target[t]-base[t],token)
    assert all(np.isfinite(v).all() for v in losses.values())
    r=dict(losses=losses,state_bytes=m.state_bytes,gate_state_bytes=gate.state_bytes,swaps=m.swaps,trials=m.trials)
    if trace:r.update(ids=np.array(ids),predictions=np.array(pred))
    return r
