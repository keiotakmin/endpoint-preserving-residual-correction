"""Low-rank residual regression with a causal, uncompressed boundary feature."""
import numpy as np
from basis_probe import columns
from channel_control import RollingControl
ARMS=['independent4','independent6','boundary4','projected4','stale4','last_residual']
class PackedRidge:
    """Independent regressions sharing only the deterministic intercept weight sum."""
    def __init__(self,k,c,p):
        self.p=p;self.decay=2.**(-1/128);self.s=np.zeros(1)
        self.cross=np.zeros((k,c,p));self.gram=np.zeros((k,c,p*(p+1)//2));self.rhs=np.zeros((k,c,p+1))
        i,j=np.triu_indices(p);self.gram[...,i==j]=1
    def predict(self,v):
        i,j=np.triu_indices(self.p);mat=np.empty((*self.cross.shape[:-1],self.p,self.p));mat[...,i,j]=self.gram;mat[...,j,i]=self.gram
        d=1+self.s[0];mat-=self.cross[..., :,None]*self.cross[...,None,:]/d
        rhs=self.rhs[...,1:]-self.cross*self.rhs[...,0,None]/d
        slope=np.linalg.solve(mat,rhs[...,None])[...,0]
        intercept=(self.rhs[...,0]-np.sum(self.cross*slope,axis=-1))/d
        return intercept+np.sum(v*slope,axis=-1)
    def update(self,v,y):
        self.s*=self.decay;self.s+=1;self.cross*=self.decay;self.cross+=v
        self.gram*=self.decay;i,j=np.triu_indices(self.p)
        self.gram+=v[...,i]*v[...,j];self.gram[...,i==j]+=1-self.decay
        self.rhs*=self.decay;self.rhs[...,0]+=y;self.rhs[...,1:]+=v*y[...,None]
    @property
    def state_bytes(self):return sum(a.nbytes for a in [self.s,self.cross,self.gram,self.rhs])

class BoundaryTracker:
    def __init__(self,h,c,arm):
        if arm not in ARMS:raise ValueError(arm)
        self.arm=arm;self.h=h
        self.uses_endpoint=arm in ['boundary4','stale4']
        if self.uses_endpoint or arm=='last_residual':self.last_endpoint=np.zeros(c)
        self.old_endpoint=np.zeros(c) if arm=='stale4' else None
        if arm=='last_residual':return
        k=6 if arm=='independent6' else 4;self.ids=np.arange(k);self.u=columns(h,self.ids);self.last=np.zeros((k,c))
        p=2 if arm.startswith('independent') else 3;self.model=PackedRidge(k,c,p)
    def issue(self,base):
        if self.arm=='last_residual':return np.broadcast_to(self.last_endpoint,base.shape),None
        features=[self.last,self.u.T@base]
        if not self.arm.startswith('independent'):
            if self.arm=='projected4':endpoint=self.u[-1]@self.last
            elif self.arm=='stale4':endpoint=self.old_endpoint
            else:endpoint=self.last_endpoint
            features.append(np.broadcast_to(np.sqrt(self.h)*endpoint,self.last.shape))
        v=np.stack(features,axis=-1);return self.u@self.model.predict(v),v
    def update(self,residual,v):
        if self.arm=='last_residual':self.last_endpoint=residual[-1].copy();return
        z=self.u.T@residual;self.model.update(v,z);self.last=z
        if self.arm=='stale4':self.old_endpoint=self.last_endpoint.copy()
        if self.uses_endpoint:self.last_endpoint=residual[-1].copy()
    @property
    def state_bytes(self):
        if self.arm=='last_residual':return self.last_endpoint.nbytes
        size=self.ids.nbytes+self.u.nbytes+self.last.nbytes+self.model.state_bytes
        if self.uses_endpoint:size+=self.last_endpoint.nbytes
        if self.old_endpoint is not None:size+=self.old_endpoint.nbytes
        return size

def evaluate(base,target,arm,bounded=False,trace=False):
    base,target=np.asarray(base,float),np.asarray(target,float)
    if base.shape!=target.shape:raise ValueError('shape mismatch')
    if bounded and (target.min()<0 or target.max()>1):raise ValueError('physical bounds')
    n,h,c=base.shape;m=BoundaryTracker(h,c,arm);g=RollingControl(c,'global','blend');losses={a:np.empty((n,c)) for a in ['static','raw','global_blend']};pred=[]
    for t in range(n):
        corr,v=m.issue(base[t]);b=base[t];raw=b+corr
        if bounded:b=np.clip(b,0,1);raw=np.clip(raw,0,1)
        corr=raw-b;issued=b+g.alpha()*corr
        if trace:pred.append(issued.copy())
        for a,p in [('static',b),('raw',raw),('global_blend',issued)]:losses[a][t]=np.mean((target[t]-p)**2,axis=0)
        g.update(corr,target[t]-b);m.update(target[t]-base[t],v)
    assert all(np.isfinite(x).all() for x in losses.values())
    r=dict(losses=losses,learner_basis_bytes=m.state_bytes,gate_bytes=g.state_bytes)
    if trace:r['predictions']=np.array(pred)
    return r
