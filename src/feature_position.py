"""Capacity-matched residual summary ablation; fixed boundary4 estimator."""
import numpy as np
from boundary_context import BoundaryTracker
from channel_control import RollingControl
ARMS=['endpoint','first','middle','mean']

class FeatureTracker(BoundaryTracker):
    def __init__(self,h,c,feature):
        if feature not in ARMS: raise ValueError(feature)
        super().__init__(h,c,'boundary4')
        self.feature=feature
    def update(self,residual,v):
        super().update(residual,v)
        if self.feature=='first': self.last_endpoint=residual[0].copy()
        elif self.feature=='middle': self.last_endpoint=residual[(self.h-1)//2].copy()
        elif self.feature=='mean': self.last_endpoint=residual.mean(axis=0)

def evaluate(base,target,feature,bounded=False,trace=False):
    base,target=np.asarray(base,float),np.asarray(target,float)
    if base.shape!=target.shape: raise ValueError('shape mismatch')
    m=FeatureTracker(base.shape[1],base.shape[2],feature)
    g=RollingControl(base.shape[2],'global','blend')
    losses={a:[] for a in ['static','raw','global_blend']};pred=[]
    for b,y in zip(base,target):
        corr,v=m.issue(b);raw=b+corr;static=np.clip(b,0,1) if bounded else b
        if bounded: raw=np.clip(raw,0,1)
        corr=raw-static;issued=static+g.alpha()*corr
        for a,p in [('static',static),('raw',raw),('global_blend',issued)]: losses[a].append(np.mean((y-p)**2,axis=0))
        if trace: pred.append(issued.copy())
        g.update(corr,y-static);m.update(y-b,v)
    result=dict(losses={a:np.asarray(v) for a,v in losses.items()},state_bytes=m.state_bytes+g.state_bytes)
    assert all(np.isfinite(x).all() for x in result['losses'].values())
    if trace: result['predictions']=np.asarray(pred)
    return result
