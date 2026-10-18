"""Regression checks for tie handling, missingness and known-truth identities."""
from pathlib import Path
import importlib.util
import sys
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod

class NumericalContracts(unittest.TestCase):
    def test_weighted_midpoint_is_invariant_to_scale(self):
        mod=load('distribution_test','src/distribution/portable_main_results.py')
        # The probability mass is exactly half at support zero, so the midpoint is one.
        for scale in [1e-6,1.0,1e6]:
            self.assertEqual(mod.wq([0,0,2],[.25*scale,.25*scale,.5*scale],.5),1.)

    def test_invalid_weights_do_not_create_observations(self):
        mod=load('distribution_test2','src/distribution/portable_main_results.py')
        self.assertEqual(mod.wq([0,2,999],[1,1,0],.5),1.)
        self.assertTrue(np.isnan(mod.wq([0,2],[0,0],.5)))

    def test_missing_value_cannot_be_replaced_with_zero(self):
        mod=load('price_test','scripts/reproduce_prices.py')
        with self.assertRaises(AssertionError):
            mod.compare(pd.DataFrame({'id':[1],'value':[0.]}),
                        pd.DataFrame({'id':[1],'value':[np.nan]}),['id'])

    def test_known_truth_reconstruction(self):
        mod=load('simulation_test','src/simulation/reconstruction.py')
        checks=mod.checks()
        self.assertEqual(checks['status'],'PASS')
        self.assertTrue(checks['zero_noise_exact_recovery'])
        self.assertLess(checks['root_cdf_max_residual'],1e-11)

if __name__=='__main__':unittest.main()
