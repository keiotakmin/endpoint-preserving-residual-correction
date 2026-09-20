"""Paper-based ELF transfer to L96/H24; see ELF_TRANSFER_PROTOCOL.md for deviations."""
import numpy as np

SEASONS={'ETTh1':24,'ETTh2':24,'ETTm1':96,'ETTm2':96,'appliances':144,'bdg2':24,'bdg2_fox':24,'bdg2_panther':24}
class FourierRidge:
 def __init__(self,l,h,c):
  self.l,self.h,self.c=l,h,c;self.ids=np.flatnonzero(np.abs(np.fft.fftfreq(l))<=.45);self.q=max(1,int(.9*h/2))
  self.G=np.zeros((c,len(self.ids),len(self.ids)),complex);self.Q=np.zeros((c,len(self.ids),self.q),complex)
  self.n=0;self.mean=np.zeros(c);self.m2=np.zeros(c);self.fits=0
 def observe(self,values):
  for row in values:
   self.n+=1;delta=row-self.mean;self.mean+=delta/self.n;self.m2+=delta*(row-self.mean)
 def sigma(self):
  s=np.sqrt(np.maximum(self.m2/max(1,self.n),0));return np.where(s>0,s,1.)
 def encode(self,x,y):
  mu=x.mean(axis=1,keepdims=True)
  a=np.fft.fft(x-mu,axis=1,norm='ortho')[:,self.ids,:].transpose(2,0,1)
  b=np.fft.rfft(y-mu,axis=1,norm='ortho')[:,:self.q,:].transpose(2,0,1)
  return a,b
 def update(self,x,y):
  a,b=self.encode(x,y);self.G+=a.conj().transpose(0,2,1)@a;self.Q+=a.conj().transpose(0,2,1)@b;self.fits+=len(x)
 def coefficients(self):
  return np.linalg.solve(self.G+20*self.sigma()[:,None,None]**2*np.eye(len(self.ids)),self.Q)
 def predict(self,x):
  mu=x.mean(axis=0);a=np.fft.fft(x-mu,axis=0,norm='ortho')[self.ids].T
  z=np.einsum('cd,cdq->cq',a,self.coefficients());p=np.zeros((self.h//2+1,self.c),complex);p[:self.q]=z.T
  return np.fft.irfft(p,n=self.h,axis=0,norm='ortho')+mu

def sigmoid(x):
 return np.exp(-np.logaddexp(0.,-x))
class ELFWeighter:
 def __init__(self,c):
  self.slow=np.zeros(c);self.merge=np.zeros(c);self.history=np.zeros((5,c));self.t=0
 def components(self):return sigmoid(-.5*self.history.sum(0)),sigmoid(self.slow),sigmoid(self.merge)
 def weight(self):
  f,s,m=self.components();return m*f+(1-m)*s
 def update(self,base,forecast,target,scale):
  f,s,_=self.components()
  loss=lambda p:np.mean(np.abs(p-target),axis=0)/scale
  delta=loss(base)-loss(forecast)
  self.merge-=.5*(loss(f*base+(1-f)*forecast)-loss(s*base+(1-s)*forecast))
  self.slow-=.5*delta;self.history[self.t%5]=delta;self.t+=1

def forecast_stream(series,origins,season):
 series=np.asarray(series,float);n=int(origins[0]);h=24;l=96;c=series.shape[1];m=FourierRidge(l,h,c);m.observe(series[n-l:n]);last=n;next_origin=n
 preds=[];scales=[];counts=[];zeros=0
 for t in origins:
  m.observe(series[last:t]);last=t
  ids=np.arange(next_origin,t-h+1)
  if len(ids):
   x=np.stack([series[s-l:s] for s in ids]);y=np.stack([series[s:s+h] for s in ids]);m.update(x,y);next_origin=int(ids[-1]+1)
  context=series[t-l:t]
  p=m.predict(context) if m.fits else np.stack([series[t-season+i%season] for i in range(h)])
  past=series[t-l-season:t];scale=np.mean(np.abs(past[season:]-past[:-season]),axis=0);zeros+=int(np.sum(scale==0));scale=np.where(scale>0,scale,1.)
  preds.append(p);scales.append(scale);counts.append(m.fits)
 return dict(predictions=np.asarray(preds),scales=np.asarray(scales),training_counts=np.asarray(counts),zero_scale_count=zeros)

def combine(base,target,forecasts,scales):
 m=ELFWeighter(base.shape[2]);pred=[];weights=[]
 for b,y,f,s in zip(base,target,forecasts,scales):
  w=m.weight() if m.t>=5 else np.ones(base.shape[2]);pred.append(w*b+(1-w)*f);weights.append(w)
  m.update(b,f,y,s)
 return np.asarray(pred),np.asarray(weights)
