"""Joint-training ablation, not a full reproduction of TEFL."""
import copy
import numpy as np
import torch
from learned_residual import Adapter
from prior_comparison_windows import split_origins

def windows(d,start,end):
 ts=split_origins(start,end)
 return tuple(torch.stack([d[t+l:t+r] for t in ts]) for l,r in [(-120,-24),(-24,0),(-96,0),(0,24)])

def forward(base,adapter,batch,cold=False):
 px,py,x,y=batch
 b=base(x)
 if adapter is None:return b
 e=py-base(px)
 if cold:
  e=e.clone();e[0]=0
 return b+adapter(e)

def score(base,adapter,data):
 base.eval()
 if adapter is not None:adapter.eval()
 total=0.;count=0
 with torch.inference_mode():
  for j in range(0,len(data[0]),32):
   batch=tuple(v[j:j+32] for v in data)
   p=forward(base,adapter,batch,cold=j==0)
   total+=float(((p-batch[-1])**2).sum());count+=p.numel()
 return total/count

def train(initial,train_data,val_data,joint):
 torch.manual_seed(0);base=copy.deepcopy(initial);adapter=Adapter(64).to(next(base.parameters()).device) if joint else None
 params=list(base.parameters())+(list(adapter.parameters()) if joint else [])
 opt=torch.optim.AdamW(params,lr=1e-3,weight_decay=.01)
 generator=torch.Generator().manual_seed(0)
 best=score(base,adapter,val_data);log=[dict(epoch=0,val_mse=best)];epoch_best=0
 def snapshot():return dict(base=copy.deepcopy(base.state_dict()),adapter=copy.deepcopy(adapter.state_dict()) if joint else None)
 state=snapshot()
 for epoch in range(1,13):
  base.train();total=0.;order=torch.randperm(len(train_data[0]),generator=generator)
  for j in range(0,len(order),32):
   ix=order[j:j+32];batch=tuple(v[ix] for v in train_data)
   opt.zero_grad();p=forward(base,adapter,batch);loss=((p-batch[-1])**2).mean();loss.backward();opt.step();total+=float(loss.detach())*len(ix)
  v=score(base,adapter,val_data);log.append(dict(epoch=epoch,train_mse=total/len(order),val_mse=v))
  if v<best:best=v;epoch_best=epoch;state=snapshot()
 base.load_state_dict(state['base'])
 if joint:adapter.load_state_dict(state['adapter'])
 assert abs(score(base,adapter,val_data)-best)<1e-7*max(1,best)
 return base,adapter,state,log,epoch_best
