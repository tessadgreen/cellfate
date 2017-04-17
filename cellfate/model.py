# Numerical solver of the differential equation
# Reference: http://scipy-cookbook.readthedocs.io/items/CoupledSpringMassSystem.html

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
import cell_density_object as cdo
from cellfate import io

test_params = [0, # k_ent
               0.05, # k_div
               0, # k_dep
               0.6, # k_bg
               0.1, # k_br
               0 # k_loss
               ]

def pd2np(input_data):
    '''
    Transforms pandas dataframe of input CellDen object into numpy array
    The resulting ndarray will have a form of (3, binDiv, binDiv, time):
    [red, green, both] x (binDiv x binDiv) x(time)
    
    Arguments:
        input_data: CellDen class object
    '''
    df = input_data.data
    binDiv = input_data.bin_num
    return np.reshape(df.as_matrix().T, (3,binDiv, binDiv, -1))

def np2pd(input_np):
    '''
    Transforms ndarray into pandas dataframe.
    The resulting pandas dataframe will have following form:
        rows: time
        columns: Red, Green, Both (in order)
            subcolumns: indices for bin (starts from 0; 
                                         in order of left to right and top to bottom)
    
    Arguments:
        input_np: ndarray of form [red, green, both] x (binDiv x binDiv) x(time)
    '''
    # Set up the column structure for dataframe
    cols=pd.MultiIndex.from_tuples([ (x,y) for x in ['R','G','Both'] \
                                    for y in np.arange(2*2)])
    reshaped = np.reshape(input_np, (12, -1))
    return pd.DataFrame(reshaped.T,columns=cols)


def diffeq(w, t, p):
    """
    Defines the differential equations for our model.

    Arguments:
        w :  vector of the state variables:
                  w = [red, grn, both]
        t :  time
        p :  vector of the parameters:
                  p = [k_ent,k_div,k_dep,k_bg,k_br, k_loss]
    """
    red, grn, both = w
    k_ent, k_div, k_dep, k_bg, k_br, k_loss = p

    # Create f = (red,grn,both) - order should be same as in w vector:
    f = [k_ent*red + (k_div - k_dep)*red + (k_br+k_loss)*both,
         k_ent*grn + (k_div - k_dep)*grn + (k_bg-k_loss)*both,
         k_ent*both + (k_div - k_dep)*both - (k_bg+k_br-k_loss)*both]
    return f

  
def diffeqSolve(params, data, stoptime=20, minStepNum=200):
    '''
    Solve the system of differential equations in diffeq() function by using
    input data as initial condition
    
    Arguments:
        params: parameter values used for diffeq system
        data: CellDen class object
        stoptime: total duration for which diffeq is solved
        minStepNum: defines minimum number of timesteps for solving diffeq
    '''
    
    data_matrix = pd2np(data)
    init_cond = data_matrix[:,:,:,0]

    # Create the time samples for the output of the ODE solver.
    nfactor = int(minStepNum/data.tot_time)+1
    numpoints = data.tot_time*nfactor

    t = np.linspace(0, stoptime, numpoints)
    
    # Create a grid to save the solved results
    binDiv = data.bin_num
    final_grid = np.zeros((3, binDiv , binDiv, data.tot_time))
    
    # Solve ODE for each bin
    for i in range(binDiv):
        for j in range(binDiv):
            wsol = odeint(diffeq, init_cond[:,i,j], t, args=(params,))
            final_grid[:,i,j,:] = wsol.T[:,::nfactor]

    return final_grid

def log_prior(theta):
    """
    returns log of prior probability distribution
    
    Parameters:
        theta: model parameters (specified as a list)
    """
    # unpack the model parameters
    k_ent, k_div, k_dep, k_bg, k_br, k_loss = theta
  
    # We can ignore normalization factor since it is constant.
    # So we simply return 0 for parameters in the specified range.
    if 0 <= k_ent <= 1 and 0 <= k_div <= 1 and 0 <= k_dep <= 1  \
    and 0 <= k_bg <= 1 and 0 <= k_br <= 1 and 0 <= k_loss <= 1:
        return 0.0
    return -np.inf
    
def log_likelihood(theta, data, sigma_n):
    """
    returns log of likelihood
    
    Parameters:
        theta: model parameters (specified as a list)
        data: CellDen class object
        sigma_n: uncertainties on measured number density
    """
    k_ent, k_div, k_dep, k_bg, k_br, k_loss = theta
    
    model = diffeqSolve(theta, data)
    data_matrix = pd2np(data)
    residual = (data_matrix - model)**2
    chi_square = np.sum(residual/(sigma_n**2))
    constant = np.sum(np.log(1/np.sqrt(2.0*np.pi*sigma_n**2)))*residual.size
                     #Need modification if sigma_n is an array
    return constant - 0.5*chi_square

def log_posterior(theta, data, sigma_n):
    """
    returns log of posterior probability distribution
    
    Parameters:
        theta: model parameters (specified as a list)
        data: CellDen class object
        sigma_n: uncertainties on measured number density
    """
    # If prior is -np.inf, no need to proceed so ends by returning -np.inf
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    
    return lp + log_likelihood(theta, data, sigma_n)
    
def plotMap(grid, duration, plotNum=10):
    '''
    Plot the heatmap of each type of cell over time
    
    Arguments:
        grid: diffeq solution arrays ([red,grn,both] x (binNum x binNum) x time)
        duration :  total timesteps
        plotNum : number of snapshots (plots). 
                  The time difference between each snapshot is even.    
    '''
    # Define time step
    time_step = int(duration/plotNum)
    red = grid[0,:,:]
    grn = grid[1,:,:]
    both = grid[2,:,:]
    # Plot heatmap for each time i*time_step
    for i in range(plotNum):
        print(i*time_step)
        print(np.shape(both))
        plt.subplot(plotNum,3,1+i*3)
        sns.heatmap(both[:,:,i*time_step], vmin=0, vmax=30, 
                         annot=True, fmt='.1f', cmap="Oranges")
        
        plt.subplot(plotNum,3,2+i*3)
        sns.heatmap(red[:,:,i*time_step], vmin=0, vmax=30, 
                         annot=True, fmt='.1f', cmap="Reds")
    
        plt.subplot(plotNum,3,3+i*3)
        sns.heatmap(grn[:,:,i*time_step], vmin=0, vmax=30, 
                         annot=True, fmt='.1f', cmap="Greens")





#######################################
# Following codes are for test purpose#
# You can ignore them                 #
#######################################

testdat = io.read('test-data.mat', 'R', 'G', 5, 2)
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

testobject = cdo.CellDen(pd.read_pickle('sample.pkl'), 5)
lpos = log_posterior(test_params, testobject, 1)
print(lpos)

#tt = diffeqSolve(test_params, testobject)
#print(np2pd(tt))

#print(df)

#plotMap(testdat_solved, 250, 3)


plt.figure()
## Plot the number of each cell over time in each bin        
#for i in range(binNum):
#    for j in range(binNum):
#        plt.subplot(4, 4, 1+i*4+j)
#        plt.xlabel('t')
#        plt.ylabel('n')
#        lw = 1
#        b = final_grid[i,j,0,:]
#        r = final_grid[i,j,1,:]
#        g = final_grid[i,j,2,:]
#        
#        plt.plot(t, b, 'orange', linewidth=lw)
#        plt.plot(t, g, 'g', linewidth=lw)
#        plt.plot(t, r, 'r', linewidth=lw)
