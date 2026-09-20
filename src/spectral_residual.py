"""MAE/SF two-stage transfer; explicit block-concatenated SF convention."""
import copy
import torch
from learned_residual import Adapter
from joint_residual import forward

def flatness(residual):
 # Adjacent nonoverlapping H-blocks form one chronological series per channel.
 x=residual.reshape(-1,residual.shape[-1]);p=torch.fft.fft(x,dim=0).abs().square()+1e-8
 return (p.log().mean(0).exp()/p.mean(0)).mean()

def score(base,adapter,data):
 base.eval();total=0.;count=0
 with torch.inference_mode():
  for j in range(0,len(data[0]),32):
   batch=tuple(v[j:j+32] for v in data);p=forward(base,adapter,batch,cold=j==0)
   total+=float((p-batch[-1]).abs().sum());count+=p.numel()
 return total/count

def warm(initial,data,sf):
 base=copy.deepcopy(initial);opt=torch.optim.AdamW(base.parameters(),lr=1e-3,weight_decay=.01);logs=[]
 for epoch in range(1,4):
  base.train();mae=0.;spectral=0.;count=0
  for j in range(0,len(data[0]),32):
   x,y=data[2][j:j+32],data[3][j:j+32];opt.zero_grad();e=y-base(x);a=e.abs().mean();s=flatness(e);loss=a+int(sf)*s;loss.backward();opt.step()
   mae+=float(a.detach())*len(x);spectral+=float(s.detach())*len(x);count+=len(x)
  logs.append(dict(epoch=epoch,train_mae=mae/count,train_sf=spectral/count))
 return base,logs

def train(initial,data,val,joint):
 torch.manual_seed(0);base=copy.deepcopy(initial);adapter=Adapter(64).to(next(base.parameters()).device) if joint else None
 opt=torch.optim.AdamW(list(base.parameters())+(list(adapter.parameters()) if joint else []),lr=1e-3,weight_decay=.01)
 gen=torch.Generator().manual_seed(0);best=score(base,adapter,val);logs=[dict(epoch=0,val_mae=best)];chosen=0
 def state():return dict(base=copy.deepcopy(base.state_dict()),adapter=copy.deepcopy(adapter.state_dict()) if joint else None)
 saved=state()
 for epoch in range(1,13):
  base.train();order=torch.randperm(len(data[0]),generator=gen);total=0.
  for j in range(0,len(order),32):
   ix=order[j:j+32];batch=tuple(v[ix] for v in data);opt.zero_grad();loss=(forward(base,adapter,batch)-batch[-1]).abs().mean();loss.backward();opt.step();total+=float(loss.detach())*len(ix)
  v=score(base,adapter,val);logs.append(dict(epoch=epoch,train_mae=total/len(order),val_mae=v))
  if v<best:best=v;chosen=epoch;saved=state()
 base.load_state_dict(saved['base'])
 if joint:adapter.load_state_dict(saved['adapter'])
 assert abs(score(base,adapter,val)-best)<1e-7
 return base,adapter,saved,logs,chosen
