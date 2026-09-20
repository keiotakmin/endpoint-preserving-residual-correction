"""Equivalent real Fourier ridge with packed symmetric statistics, zero DC removed."""
import numpy as np
from elf_transfer import FourierRidge
from elf_capacity import CONFIGS
class CompactRidge(FourierRidge):
 def __init__(self,l,h,c,d,q):
  self.l,self.h,self.c,self.q=l,h,c,q;self.ids=np.arange(1,(d-1)//2+1);self.r=d-1
  self.G=np.zeros((c,self.r*(self.r+1)//2));self.Q=np.zeros((c,self.r,2*q-1));self.n=0;self.mean=np.zeros(c);self.m2=np.zeros(c);self.fits=0
 def encode(self,x,y):
  mu=x.mean(1,keepdims=True);a=np.fft.rfft(x-mu,axis=1,norm='ortho')[:,self.ids,:].transpose(2,0,1)
  a=np.sqrt(2)*np.concatenate([a.real,a.imag],axis=-1)
  b=np.fft.rfft(y-mu,axis=1,norm='ortho')[:,:self.q,:].transpose(2,0,1)
  return a,np.concatenate([b.real,b.imag[...,1:]],axis=-1)
 def update(self,x,y):
  a,b=self.encode(x,y);i,j=np.triu_indices(self.r);self.G+=(a.transpose(0,2,1)@a)[:,i,j];self.Q+=a.transpose(0,2,1)@b;self.fits+=len(x)
 def coefficients(self):
  i,j=np.triu_indices(self.r);g=np.empty((self.c,self.r,self.r));g[:,i,j]=self.G;g[:,j,i]=self.G
  return np.linalg.solve(g+20*self.sigma()[:,None,None]**2*np.eye(self.r),self.Q)
 def predict(self,x):
  mu=x.mean(0);a=np.fft.rfft(x-mu,axis=0,norm='ortho')[self.ids].T;a=np.sqrt(2)*np.concatenate([a.real,a.imag],axis=-1)
  z=np.einsum('cd,cdq->cq',a,self.coefficients());p=np.zeros((self.h//2+1,self.c),complex);p[:self.q]=z[:,:self.q].T;p[1:self.q]+=1j*z[:,self.q:].T
  return np.fft.irfft(p,n=self.h,axis=0,norm='ortho')+mu
 @property
 def state_bytes(self):return sum(x.nbytes for x in vars(self).values() if isinstance(x,np.ndarray))

def forecast_stream(series,origins,season,d=87,q=10):
 stride=24
 if stride not in (1,24):raise ValueError(stride)
 series=np.asarray(series,float);n=int(origins[0]);h=24;l=96;c=series.shape[1];m=CompactRidge(l,h,c,d,q);m.observe(series[n-l:n]);last=n;next_origin=n
 preds=[];scales=[];counts=[];zeros=0
 for t in origins:
  m.observe(series[last:t]);last=t
  ids=np.arange(next_origin,t-h+1,stride)
  if len(ids):
   x=np.stack([series[s-l:s] for s in ids]);y=np.stack([series[s:s+h] for s in ids]);m.update(x,y);next_origin=int(ids[-1]+stride)
  context=series[t-l:t]
  p=m.predict(context) if m.fits else np.stack([series[t-season+i%season] for i in range(h)])
  past=series[t-l-season:t];scale=np.mean(np.abs(past[season:]-past[:-season]),axis=0);zeros+=int(np.sum(scale==0));scale=np.where(scale>0,scale,1.)
  preds.append(p);scales.append(scale);counts.append(m.fits)
 return dict(predictions=np.asarray(preds),scales=np.asarray(scales),training_counts=np.asarray(counts),zero_scale_count=zeros)
