"""Exponentially weighted ridge regression with a persistent ridge penalty."""
import numpy as np
from residual_tracking import basis
from channel_control import RollingControl
ARMS={'growing':None,'half32':32,'half128':128,'half512':512}

class WeightedRidge:
    def __init__(self,shape,features=3,half_life=None):
        if half_life is not None and half_life<=0: raise ValueError('positive half life required')
        self.decay=1. if half_life is None else 2.**(-1./half_life)
        self.G=np.broadcast_to(np.eye(features),(*shape,features,features)).copy()
        self.q=np.zeros((*shape,features))
    def predict(self,x):
        beta=np.linalg.solve(self.G,self.q[...,None])[...,0]
        return np.sum(x*beta,axis=-1)
    def update(self,x,y):
        self.G*=self.decay
        idx=np.arange(self.G.shape[-1]);self.G[...,idx,idx]+=1-self.decay
        self.G+=np.einsum('...i,...j->...ij',x,x)
        self.q*=self.decay;self.q+=x*y[...,None]
    @property
    def state_bytes(self):return self.G.nbytes+self.q.nbytes

def evaluate(base,target,half_life=None,bounded=False,trace=False):
    base,target=np.asarray(base,float),np.asarray(target,float)
    if bounded and (target.min()<0 or target.max()>1):raise ValueError('targets outside [0,1]')
    n,h,c=base.shape;u=basis(h);last=np.zeros((3,c));model=WeightedRidge((3,c),half_life=half_life)
    gate=RollingControl(c,'global','blend')
    losses={p:np.empty((n,c)) for p in ['static','raw','global_blend']};alphas=np.empty(n);preds=[]
    for t in range(n):
        x=np.stack([np.ones((3,c)),last,u.T@base[t]],axis=-1)
        b=base[t];p=b+u@model.predict(x)
        if bounded:b=np.clip(b,0,1);p=np.clip(p,0,1)
        alpha=gate.alpha();corr=p-b;issued=b+alpha*corr
        if trace:preds.append(issued.copy())
        alphas[t]=alpha[0]
        for name,forecast in [('static',b),('raw',p),('global_blend',issued)]:
            losses[name][t]=np.mean((target[t]-forecast)**2,axis=0)
        gate.update(corr,target[t]-b)
        last=u.T@(target[t]-base[t]);model.update(x,last)
    assert all(np.isfinite(v).all() for v in losses.values())
    result=dict(losses=losses,alpha=alphas,learner_state_bytes=model.state_bytes+last.nbytes,gate_state_bytes=gate.state_bytes)
    if trace:result['predictions']=np.array(preds)
    return result
