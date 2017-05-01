from unittest import TestCase
import numpy as np
import pandas as pd
import cellfate
from cellfate import io
from cellfate import model 
from cellfate import celldensity
import pickle as pkl
from cellfate import model
import scipy.optimize as op


class Test(TestCase):
    def test_is_string(self):
        """Basic Test of test_print"""
        s = cellfate.test_print()
        self.assertTrue(isinstance(s, str))

    def test_io(self):
        """ Test of data import"""
        x = io.read('io-test.csv','Sox2','Oct4',2)
        #check that there are 28 cells total
        assert x.data.sum().sum()==28.0
        assert x.bin_num==2
        assert (x.data['Sox2'][3].values==[2.,1.,3.]).all()
        
    def test_diffeq_solver(self):
        """Tests that test_diffeq_solver returns array of correct shape"""
        nbins=6
        testdat=io.read('io-test.csv','Sox2','Oct4',nbins)
        test_params=[0.05, 0.6, 0.1]
        testdat_solved = model.diffeqSolve(test_params, testdat)
        assert np.shape(testdat_solved)==(3,nbins,nbins,3)
    
    def test_model(self):
        """Tests that likelihood function returns a value"""
        test_params=[0.03,0.1,0.7]
        nbins=4
        testdat=io.read('io-test.csv','Sox2','Oct4',nbins)
        sigma_n=0.2
        val=model.log_likelihood(test_params, testdat, sigma_n)
        assert isinstance(val,float)

    def test_model_regress(self):
        """ Tests that model is running correctly"""

        # Parameters to calculate likelihood function
        params1 = [0.05, 0.1, 0.6]
        params2 = [0.1, 0.2, 0.4]

        nbins=4
        testdat=io.read('io-test.csv','Sox2','Oct4',nbins)

        # Calculate log_likelihood function
        # sigma_n is set arbitrarily as 0.4
        val_1 = model.log_likelihood(params1, testdat, 0.4)
        val_2 = model.log_likelihood(params2, testdat, 0.4)

        self.assertTrue(val_2 < val_1)

    def test_inference(self):
        """ Tests that inference on simulated data returns accurate params """

        #import data 
        path=io.get_data_file_path('simulated_data.pkl')
        test_data=pd.read_pickle(path)
        data=celldensity.CellDen(test_data)
        #test data generated using k=[0.018, 0.001, 0.002]

        k0=[0.02,0.005,0.001]
        res = op.fmin(model.negative_log_posterior, k0, args=(data, 1000))
        np.testing.assert_almost_equal(res,k0,2)


        #sampler=model.k_sampler(data, 1000, res, 1e-4)
        #samples = sampler.chain[:,300:,:]
