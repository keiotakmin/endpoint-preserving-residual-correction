import unittest
import numpy as np
from elf_real import CompactRidge
from elf_capacity import CompactRidge as Original,CONFIGS
class Tests(unittest.TestCase):
 def test_equivalence_and_bytes(self):
  rng=np.random.default_rng(12)
  for d,q in CONFIGS.values():
   a=Original(96,24,3,d,q);b=CompactRidge(96,24,3,d,q);obs=rng.normal(size=(96,3));a.observe(obs);b.observe(obs)
   for _ in range(3):
    x=rng.normal(size=(8,96,3));y=rng.normal(size=(8,24,3));a.update(x,y);b.update(x,y)
    np.testing.assert_allclose(a.predict(x[0]),b.predict(x[0]),atol=1e-12,rtol=1e-12)
   r=d-1;self.assertEqual(b.state_bytes,8*(3*(r*(r+1)//2+r*(2*q-1)+2)+(d-1)//2));self.assertLess(b.state_bytes,a.state_bytes)
 def test_direct_real_solve(self):
  rng=np.random.default_rng(8);m=CompactRidge(96,24,2,9,4);x=rng.normal(size=(10,96,2));y=rng.normal(size=(10,24,2));m.observe(x[0]);a,b=m.encode(x,y);m.update(x,y)
  expected=np.linalg.solve(a.transpose(0,2,1)@a+20*m.sigma()[:,None,None]**2*np.eye(8),a.transpose(0,2,1)@b)
  np.testing.assert_allclose(expected,m.coefficients(),atol=1e-12)
if __name__=='__main__':unittest.main()
