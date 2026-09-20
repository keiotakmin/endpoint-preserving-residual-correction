"""Prespecified Fourier input/output ablations, fixed sparse ridge20."""
import numpy as np
from elf_transfer import FourierRidge
CONFIGS={'full':(87,10),'in3':(3,10),'in5':(5,10),'in9':(9,10),'out4':(87,4),'compact3':(3,4),'compact5':(5,4)}
class CompactRidge(FourierRidge):
 def __init__(self,l,h,c,d,q):
  super().__init__(l,h,c)
  if d!=len(self.ids):
   self.ids=np.flatnonzero(np.abs(np.fft.fftfreq(l)*l)<=(d-1)//2)
  assert len(self.ids)==d
  self.q=q;self.G=np.zeros((c,d,d),complex);self.Q=np.zeros((c,d,q),complex)
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
