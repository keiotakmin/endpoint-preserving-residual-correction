"""Input safeguards and prediction-before-update tests for the portable runners."""
import hashlib
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from prepare_data import prepare
from evaluate import blended_losses


class PipelineTests(unittest.TestCase):
    def test_order_and_hash_reject_changed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);raw=root/'raw.csv';out=root/'out.csv'
            pd.DataFrame({'date':['a','b'],'second':[2.,3.],'first':[5.,6.],'rv1':[99.,99.]}).to_csv(raw,index=False)
            values=np.array([[2.,5.],[3.,6.]],dtype='<f4')
            spec=dict(columns=['second','first'],rows=2,float32_sha256=hashlib.sha256(values.tobytes()).hexdigest(),url='test')
            prepare('test',spec,raw,out)
            with self.assertRaises(ValueError):prepare('test',dict(spec,columns=['first','second']),raw,out)
            changed=pd.read_csv(raw);changed.loc[1,'second']=100.;changed.to_csv(raw,index=False)
            with self.assertRaises(ValueError):prepare('test',spec,raw,out)

    def test_controller_does_not_use_current_target(self):
        base=np.zeros((4,24,2));raw=np.ones_like(base);target=np.ones_like(base)
        loss=blended_losses(base,target,raw)
        self.assertEqual(loss[0,1,0],1.)  # No completed target means zero blend.
        altered=target.copy();altered[2:]=17.
        np.testing.assert_array_equal(blended_losses(base,altered,raw)[:2],loss[:2])
        self.assertEqual(loss[1,1,0],0.)


if __name__=='__main__':unittest.main()
