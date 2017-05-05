from unittest import TestCase
import numpy as np
import pandas as pd
import cellfate
from cellfate import io, model, celldensity
import scipy.optimize as op


class Test(TestCase):
    def test_is_string(self):
        """Basic Test of test_print"""
        s = cellfate.test_print()
        self.assertTrue(isinstance(s, str))

    # io.py
    def test_io(self):
        """ Test of data import"""
        x = io.read('io-test.csv','Sox2','Oct4',2)
        #check that there are 28 cells total
        assert x.data.sum().sum()==28.0
        assert x.bin_num==2
        assert (x.data['Sox2'][3].values==[2.,1.,3.]).all()


    # model.py
    def test_solver_uncoupled(self):
        """Tests that solver_uncoupled function returns array of correct shape and value"""
        path=io.get_data_file_path('simulated_uncoupled_6x6.pkl')
        testdata=celldensity.CellDen(pd.read_pickle(path))
        test_params=[0.03,  0.005,  0.005]

        testdata_matrix = testdata.pd2np()
        testdata_solved = model.solver_uncoupled(test_params, testdata)
        assert np.shape(testdata_solved)==np.shape(testdata_matrix)
        np.testing.assert_almost_equal(testdata_solved[0,1,2,3],testdata_matrix[0,1,2,3],3)

    def test_log_prior_uncoupled(self):
        """Tests that log_prior_uncoupled function returns a correct value"""
        self.assertTrue(model.log_prior_uncoupled([0.5, 0.5, 0.5])==0)
        self.assertTrue(model.log_prior_uncoupled([2, 0.5, 0.5])==-np.inf)

    def test_log_likelihood_uncoupled(self):
        """Tests that log_likelihood_uncoupled function returns a reasonable estimate"""
        test_params=[ 0.03966004,  0.00523172,  0.00523965]

        path=io.get_data_file_path('simulated_uncoupled_6x6.pkl')
        testdata=celldensity.CellDen(pd.read_pickle(path))
        mu_n = -0.15
        sigma_n=0.1

        val=model.log_likelihood_uncoupled(test_params, testdata, mu_n, sigma_n)
        np.testing.assert_almost_equal(val,-10410.36855,4)
        
    def test_log_posterior_uncoupled(self):
        """Tests that log_posterior_uncoupled function returns a reasonable estimate"""
        test_params=[ 0.03966004,  0.00523172,  0.00523965]

        path=io.get_data_file_path('simulated_uncoupled_6x6.pkl')
        testdata=celldensity.CellDen(pd.read_pickle(path))
        mu_n = -0.15
        sigma_n=0.1
        
        lp = model.log_prior_uncoupled(test_params)
        ll=model.log_likelihood_uncoupled(test_params, testdata, mu_n, sigma_n)
        self.assertTrue((ll+lp)==model.log_posterior_uncoupled(test_params, testdata, mu_n, sigma_n))

    def test_negative_log_posterior_uncoupled(self):
        """Tests that negative_log_posterior_uncoupled function returns correct value"""
        test_params=[ 0.03966004,  0.00523172,  0.00523965]

        path=io.get_data_file_path('simulated_uncoupled_6x6.pkl')
        testdata=celldensity.CellDen(pd.read_pickle(path))
        mu_n = -0.15
        sigma_n=0.1
        
        ll=model.log_likelihood_uncoupled(test_params, testdata, mu_n, sigma_n)
        self.assertTrue((-1*ll)==model.negative_log_posterior_uncoupled(test_params, testdata, mu_n, sigma_n))

    def test_solver_coupled(self):
        """Tests that solver_coupled function returns array of correct shape and value"""
        path=io.get_data_file_path('simulated_coupled_6x6.pkl')
        testdata=celldensity.CellDen(pd.read_pickle(path))
        test_params=[0.022,0.003,0.00016,0.00022]

        testdata_matrix = testdata.pd2np()[:,1:-1,1:-1,:]
        testdata_solved = model.solver_coupled(test_params, testdata)[:,1:-1,1:-1,:]
        assert np.shape(testdata_solved)==np.shape(testdata_matrix)
        np.testing.assert_almost_equal(testdata_solved[0,1,2,3],testdata_matrix[0,1,2,3],3)

    def test_log_prior_coupled(self):
        """Tests that log_prior_coupled function returns a correct value"""
        self.assertTrue(model.log_prior_coupled([0.5, 0.5, 0.5, 0.5])==0)
        self.assertTrue(model.log_prior_coupled([2, 0.5, 0.5, 0.5])==-np.inf)

    def test_log_likelihood_coupled(self):
        """Tests that log_likelihood_coupled function returns a reasonable estimate"""
        test_params=[0.03178564,  0.00310762,  0.00017541,  0.00022762]

        path=io.get_data_file_path('simulated_coupled_6x6.pkl')
        testdata=celldensity.CellDen(pd.read_pickle(path))
        mu_n = -0.15
        sigma_n=0.1

        val=model.log_likelihood_coupled(test_params, testdata, mu_n, sigma_n)
        np.testing.assert_almost_equal(val,-10289.069087654105,4)

    def test_log_posterior_coupled(self):
        """Tests that log_posterior_coupled function returns a reasonable estimate"""
        test_params=[0.03178564,  0.00310762,  0.00017541,  0.00022762]

        path=io.get_data_file_path('simulated_coupled_6x6.pkl')
        testdata=celldensity.CellDen(pd.read_pickle(path))
        mu_n = -0.15
        sigma_n=0.1
        
        lp = model.log_prior_coupled(test_params)
        ll=model.log_likelihood_coupled(test_params, testdata, mu_n, sigma_n)
        self.assertTrue((ll+lp)==model.log_posterior_coupled(test_params, testdata, mu_n, sigma_n))
    
    def test_residual(self):
        """Tests that residual function returns a correct value"""
        params1 = [0.1, 0.1, 0.1]
        params2 = [0.1, 0.1, 0.1, 0.1]

        path=io.get_data_file_path('simulated_coupled_6x6.pkl')
        testdata=celldensity.CellDen(pd.read_pickle(path))
        mu_n = -0.15
        sigma_n=0.1
        
        chi2_uncoupled = np.sum(model.residual(params1, testdata, mu_n, sigma_n, False)**2)
        chi2_coupled = np.sum(model.residual(params2, testdata, mu_n, sigma_n, True)**2)

        np.testing.assert_almost_equal(1604132.5558146131, chi2_uncoupled, 5)        
        np.testing.assert_almost_equal(2277944.6229283987, chi2_coupled, 5)
    

    def test_model_regress(self):
        """ Tests that model is running correctly"""

        # Parameters to calculate likelihood function
        params1 = [0.04, 0.005, 0.005]
        params2 = [0.1, 0.1, 0.1]

        path=io.get_data_file_path('simulated_uncoupled_6x6.pkl')
        testdata=celldensity.CellDen(pd.read_pickle(path))

        # Calculate log_likelihood function
        # sigma_n is set arbitrarily as 0.4
        val_1 = model.log_likelihood_uncoupled(params1, testdata, -0.15, 0.1)
        val_2 = model.log_likelihood_uncoupled(params2, testdata, -0.15, 0.1)

        self.assertTrue(val_2 < val_1)

    def test_inference(self):
        """ Tests that inference on simulated data returns accurate params """

        #import data 
        path=io.get_data_file_path('simulated_data.pkl')
        test_data=pd.read_pickle(path)
        data=celldensity.CellDen(test_data)
        #test data generated using k=[0.018, 0.001, 0.002]

        k0=[0.02,0.005,0.001]
        res = op.fmin(model.negative_log_posterior_uncoupled, k0, args=(data, -0.15, 0.1))
        np.testing.assert_almost_equal(res,k0,2)
