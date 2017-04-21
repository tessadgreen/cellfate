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
        x = io.read('test-data','R','G',5,10)
        assert isinstance(x.cellwidth,int)
        assert x.bin_num==10

    def test_csv_io(self):
        x = io.read_csv('io-test.csv','Oct4','Sox2',5)

    def test_diffeq_solver(self):
        nbins=6
        testdat=io.read('test-data','R','G', 5,nbins)
        test_params=[0.05, 0.6, 0.1]
        #test_params = [0, # k_ent
        #       0.05, # k_div
        #       0, # k_dep
        #       0.6, # k_bg
        #       0.1, # k_br
        #       0 # k_loss
        #       ]
        testdat_solved = model.diffeqSolve(test_params, testdat)
        assert np.shape(testdat_solved)==(3,nbins,nbins,1)
    
    def test_model(self):
        #test_params = [0, # k_ent
        #    0.03, # k_div
        #    0, # k_dep
        #    0.1, # k_bg
        #    0.7, # k_br
        #    0 # k_loss
        #       ]
        test_params=[0.03,0.1,0.7]
        nbins=4
        testdat=io.read('test-data','R','G', 5,nbins)
        sigma_n=0.2
        val=model.log_likelihood(test_params, testdat, sigma_n)
        assert isinstance(val,float)


#testdat = io.read('test-data.mat', 'R', 'G', 5, 2)
#testdat_grid = np.reshape(testdat.data.as_matrix(), (3,testdat.bin_num,testdat.bin_num))
#testdat_solved = diffeqSolve(test_params, testdat_grid)
#params = [0, 0.1, 0, 0.6, 0.2, 0]
#
#
#cols=pd.MultiIndex.from_tuples([ (x,y) for x in ['R','G','Both'] for y in np.arange(2*2)])
#reshaped = np.reshape(testdat_solved, (12, -1))
#df = pd.DataFrame(reshaped.T,columns=cols)
#df.to_pickle('sample.pkl')
#print(np.size(testdat_grid))

#testobject = cdo.CellDen(pd.read_pickle('sample.pkl'), 5)
#lpos = log_posterior(test_params, testobject, 1)
#print(lpos)

#tt = diffeqSolve(test_params, testobject)
#print(np2pd(tt))

#print(df)