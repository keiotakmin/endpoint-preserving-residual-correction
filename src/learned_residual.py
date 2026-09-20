"""Frozen-base, offline-trained residual adapter; not the full TEFL protocol."""
import copy
import numpy as np
import torch
from torch import nn

class Adapter(nn.Module):
 def __init__(self,rank):
  super().__init__();self.down=nn.Linear(24,rank,bias=False);self.up=nn.Linear(rank,24,bias=False);nn.init.zeros_(self.up.weight)
 def forward(self,x):return self.up(torch.relu(self.down(x.transpose(-1,-2)))).transpose(-1,-2)

def train_adapter(train_x,train_y,val_x,val_y,rank,seed):
 torch.manual_seed(seed);m=Adapter(rank);opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=.01)
 x=torch.as_tensor(train_x,dtype=torch.float32);y=torch.as_tensor(train_y,dtype=torch.float32);vx=torch.as_tensor(val_x,dtype=torch.float32);vy=torch.as_tensor(val_y,dtype=torch.float32)
 with torch.no_grad():best=float(torch.mean((m(vx)-vy)**2))
 state=copy.deepcopy(m.state_dict());best_epoch=0;log=[dict(epoch=0,val_mse=best)]
 for epoch in range(1,13):
  order=torch.randperm(len(x));total=0.
  for j in range(0,len(x),32):
   ix=order[j:j+32];opt.zero_grad();loss=torch.mean((m(x[ix])-y[ix])**2);loss.backward();opt.step();total+=float(loss.detach())*len(ix)
  with torch.no_grad():score=float(torch.mean((m(vx)-vy)**2))
  log.append(dict(epoch=epoch,train_mse=total/len(x),val_mse=score))
  if score<best:best=score;best_epoch=epoch;state=copy.deepcopy(m.state_dict())
 m.load_state_dict(state);m.eval();return m,log,best_epoch

def predict(model,base,target):
 # Every residual at index t is only used for the prediction at t+1.
 previous=np.zeros_like(base,dtype=np.float32);previous[1:]=(target[:-1]-base[:-1]).astype(np.float32)
 with torch.inference_mode():corr=model(torch.from_numpy(previous)).numpy()
 return base+corr
