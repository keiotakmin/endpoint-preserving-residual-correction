import unittest
import numpy as np
import torch
from spectral_residual import flatness
class Tests(unittest.TestCase):
 def test_formula_and_gradient(self):
  x=torch.randn(4,24,3,dtype=torch.float64,requires_grad=True);p=np.abs(np.fft.fft(x.detach().numpy().reshape(-1,3),axis=0))**2+1e-8
  ref=np.mean(np.exp(np.log(p).mean(0))/p.mean(0));self.assertAlmostEqual(float(flatness(x)),ref,places=12)
  flatness(x).backward();self.assertTrue(torch.isfinite(x.grad).all());self.assertGreater(float(x.grad.abs().sum()),0)
 def test_structured_and_impulse(self):
  t=torch.arange(96,dtype=torch.float64);s=torch.sin(2*torch.pi*t/12).reshape(4,24,1);i=torch.zeros_like(s);i[0,0]=1
  self.assertLess(float(flatness(s)),1e-5);self.assertAlmostEqual(float(flatness(i)),1,places=12)
  z=torch.zeros_like(s,requires_grad=True);flatness(z).backward();self.assertTrue(torch.isfinite(z.grad).all())
if __name__=='__main__':unittest.main()
