from unittest import TestCase
import numpy as np
import cellfate
from cellfate import io
from cellfate import model 

class Test(TestCase):
    def test_is_string(self):
        s = cellfate.test_print()
        self.assertTrue(isinstance(s, str))

    def test_io(self):
        x = io.read('io-test.csv','Sox2','Oct4',2)
        assert x.bin_num==2

    def test_diffeq_solver(self):
        nbins=6
        testdat=io.read('io-test.csv','Sox2','Oct4',nbins)
        test_params=[0.05, 0.6, 0.1]
        testdat_solved = model.diffeqSolve(test_params, testdat)
        assert np.shape(testdat_solved)==(3,nbins,nbins,3)
    
    def test_model(self):
        test_params=[0.03,0.1,0.7]
        nbins=4
        testdat=io.read('io-test.csv','Sox2','Oct4',nbins)
        sigma_n=0.2
        val=model.log_likelihood(test_params, testdat, sigma_n)
        assert isinstance(val,float)