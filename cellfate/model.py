# Numerical solver of the differential equation
# Reference: 
# 1. http://scipy-cookbook.readthedocs.io/items/CoupledSpringMassSystem.html
# 2. http://ipython-books.github.io/featured-05/

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import emcee
import pandas as pd

def pd2np(input_data):
    '''
    Transforms data of input CellDen object ( inpandas Dataframe) into numpy array
    and returns the numpy array.
    The resulting ndarray will have a form of (3, binDiv, binDiv, time):
    [red, green, both] x (binDiv x binDiv) x(time)
    
    Arguments:
        input_data: CellDen class object
    '''
    df = input_data.data
    binDiv = input_data.bin_num
    return np.reshape(df.as_matrix().T, (3,binDiv, binDiv, -1))

def np2pd(input_np, binNum):
    '''
    Transforms ndarray into pandas dataframe and returns the dataframe.
    The resulting pandas dataframe will have following form:
        rows: time
        columns (in order): Sox2 (green), Oct4 (red), Both
            subcolumns: indices for bin (starts from 0; in the order of 
                                         left to right and top to bottom)
    
    Arguments:
        input_np: ndarray of form (3, binDiv, binDiv, time)
                  i.e. [Sox2, Oct4, Both] x (binDiv x binDiv) x(time)
    '''
    # Set up the column structure for dataframe
    cols=pd.MultiIndex.from_tuples([ (x,y) for x in ['Sox2','Oct4','Both'] \
                                    for y in np.arange(binNum*binNum)])
    reshaped = np.reshape(input_np, (3*binNum**2, -1))
    return pd.DataFrame(reshaped.T,columns=cols)

  
def solver_uncoupled(params, data, minStepNum=200):
    '''
    Solve the system of differential equations in model_uncouple function by using
    input data as initial condition.
    Returns solution in numpy ndarray in the form of (3, BinDiv, BinDiv, time)
    
    Arguments:
        params: parameter values used for diffeq system
        data: CellDen class object
        minStepNum: defines minimum number of timesteps for solving diffeq
    '''
    
    data_matrix = data.pd2np()
    init_cond = data_matrix[:,:,:,0]
    stepNum = data.tot_time
    stoptime = data.time_scale*stepNum # total time
    
    def model_uncoupled(w, t, p):
        """
        Returns the system of differential equations defining the model without diffusion.
    
        Arguments:
            w :  vector of the state variables:
                     w = [grn, red, both]
            t :  time
            p :  vector of the parameters:
                      p = [k_div,k_bg,k_br]
        """
        grn, red, both = w
        k_div, k_bg, k_br = p
    
        # Create f = (red,grn,both) - order should be same as in theta vector:
        f = [k_div*grn + k_bg*both,
             k_div*red + k_br*both,
             k_div*both - (k_bg+k_br)*both]
    
        return f

    # Create the time samples for the output of the ODE solver.
    nfactor = int(minStepNum/stepNum)+1
    numpoints = stepNum*nfactor

    t = np.linspace(0, stoptime, numpoints)
    
    # Create a grid to save the solved results
    binDiv = data.bin_num
    final_grid = np.zeros((3, binDiv , binDiv, stepNum))
    # Solve ODE for each bin
    for i in range(binDiv):
        for j in range(binDiv):
            wsol = odeint(model_uncoupled, init_cond[:,i,j], t, args=(params,))
            final_grid[:,i,j,:] = wsol.T[:,::nfactor]

    return final_grid


def log_prior_uncoupled(theta):
    """
    Returns log of prior probability distribution for the model without diffusion
    
    Arguments:
        theta: model parameters (specified as a list)
                theta = [k_div, k_bg, k_br]
        
    """
    # unpack the model parameters
    k_div, k_bg, k_br = theta
  
    # We can ignore normalization factor since it is constant.
    # So we simply return 0 for parameters in the specified range.
    if 0 <= k_div <= 1 and 0 <= k_bg <= 1 and 0 <= k_br <= 1:
        return 0.0
    return -np.inf
    
def log_likelihood_uncoupled(theta, data, mu_n, sigma_n):
    """
    Returns log of likelihood function for model without diffusion.
    
    Parameters:
        theta: model parameters (specified as a list)
                theta = [k_div, k_bg, k_br]
        data: CellDen class object
        mu_n: mean of the log distribution of counting error
        sigma_n: standard deviation of the log distribution of counting error    
    """
    model = solver_uncoupled(theta, data)
    data_matrix = data.pd2np()

    # Remove the bins in whcih observed number of cells is zero
    nonzero_args = np.nonzero(data_matrix)
    data_matrix = data_matrix[nonzero_args]
    model = model[nonzero_args]

    # Remove the bins in whcih number of cells of the model is zero
    # These zero points become problems as we take log of them.
    # While it cannot be rigorously justified, the number of such case is
    # approximately less than 5%, so we expect contribution of these points
    # is negligible
    nonzero_args = np.nonzero(model)
    data_matrix = data_matrix[nonzero_args]
    model = model[nonzero_args]
            
    residual = (np.log(data_matrix) - np.log(model) - mu_n)**2
    chi_square = np.sum(residual/(sigma_n**2))
    constant = np.sum(np.log(1/np.sqrt(2.0*np.pi*sigma_n**2)))*residual.size
    logsum_data = np.sum(np.log(data_matrix))
                     
    return constant - logsum_data - 0.5*chi_square

def log_posterior_uncoupled(theta, data, mu_n, sigma_n):
    '''
    Returns log of posterior probability distribution for model without diffusion.
    
    Parameters:
        theta: model parameters (specified as a list)
                theta = [k_div, k_bg, k_br]
        data: CellDen class object
        mu_n: mean of the log distribution of counting error
        sigma_n: standard deviation of the log distribution of counting error    
   '''
    # If prior is -np.inf, no need to proceed so ends by returning -np.inf
    lp = log_prior_uncoupled(theta)
    if not np.isfinite(lp):
        return -np.inf
    
    return lp + log_likelihood_uncoupled(theta, data, mu_n, sigma_n)

def negative_log_posterior_uncoupled(theta, data, mu_n, sigma_n):
    '''
    Returns the negative value of log_posterior_uncoupled function.
    
    Arguments:
        theta: model parameters (specified as a list)
                theta = [k_div, k_bg, k_br]
        data: CellDen class object
        mu_n: mean of the log distribution of counting error
        sigma_n: standard deviation of the log distribution of counting error    
    '''
    return -log_posterior_uncoupled(theta, data, mu_n, sigma_n)


def solver_coupled(theta, data, minStepNum=200):
    '''
    Solves the system of differential equations for the model with diffusion by
    using input data as initial condition.
    It uses finite difference method to solve the system of PDEs.
    Returns solution in numpy ndarray in the form of (3, BinDiv, BinDiv, time)
    
    Arguments:
        theta: model parameters (specified as a list)
                theta = [k_div, k_bg, k_br, D]
        data: CellDen class object
        minStepNum: defines minimum number of timesteps for solving diffeq
    '''
    # Define variables
    data_matrix = data.pd2np() # data
    init_cond = data_matrix[:,:,:,0] # initial condition
    k_div, k_bg, k_br, k_mov = theta # parameters
    
    binDiv = data.bin_num # bin number (one side)
    length = data.length_scale # length of the side of the area (mm)
    dx = length/binDiv  # space step

    stepNum = data.tot_time    
    T = data.time_scale*stepNum  # total time
    dt = T/stepNum # time step
    
    dt_cutoff = 0.9 * dx**2/2 # Stability criterion
    if dt > dt_cutoff:
        nfactor = int(dt/dt_cutoff)+1
        stepNum = nfactor*stepNum
        dt = T/stepNum
    else:
        nfactor = 1
    
    B = init_cond[2,:,:] # Both cells
    R = init_cond[1,:,:] # Oct4 (red) cells
    G = init_cond[0,:,:] # Sox2 (green) cells

    def laplacian(Z):
        '''
        Returns laplacian using finite difference method.
        Note that the edge is removed.
        
        Arguments:
            Z: 2-d numpy array of number density
        '''
        Ztop = Z[0:-2,1:-1]
        Zleft = Z[1:-1,0:-2]
        Zbottom = Z[2:,1:-1]
        Zright = Z[1:-1,2:]
        Zcenter = Z[1:-1,1:-1]
        return (Ztop + Zleft + Zbottom + Zright - 4 * Zcenter) / dx**2

    # We simulate the PDE with the finite difference method.
    # Create a grid to save the solution over time
    final_grid = np.zeros((3, binDiv , binDiv, data.tot_time))
    final_grid[:,:,:,0] = init_cond

    for i in range(stepNum-1):
        # We compute the Laplacian of B,R,G.
        deltaB = laplacian(B)
        deltaR = laplacian(R)
        deltaG = laplacian(G)
        # We take the values of B,R,G inside the grid. Again, edge is removed
        Bc = B[1:-1,1:-1]
        Rc = R[1:-1,1:-1]
        Gc = G[1:-1,1:-1]
        # We update the variables.
        B[1:-1,1:-1] = Bc + dt * (k_mov * deltaB + k_div*Bc - (k_bg+k_br)*Bc)
        R[1:-1,1:-1] = Rc + dt * (k_mov * deltaR + k_div*Rc + k_br*Bc)
        G[1:-1,1:-1] = Gc + dt * (k_mov * deltaG + k_div*Gc + k_bg*Bc)
            
        # Neumann conditions: derivatives at the edges are null
        for Z in [B, R, G]:
            Z[0,:] = Z[1,:]
            Z[-1,:] = Z[-2,:]
            Z[:,0] = Z[:,1]
            Z[:,-1] = Z[:,-2]
        
        # Save the result at this step
        # If nfactor > 1, save the result at every nfactor-th step
        if ((i+1)%nfactor)==0:
            index = int((i+1)/nfactor)
            final_grid[0,:,:,index] = G
            final_grid[1,:,:,index] = R
            final_grid[2,:,:,index] = B
    
    return final_grid


def log_prior_coupled(theta):
    '''
    Returns log of prior probability distribution for the model with diffusion.
    
    Arguments:
        theta: model parameters (specified as a list)
                theta = [k_div, k_bg, k_br, k_mov]        
    '''
    # unpack the model parameters
    k_div, k_bg, k_br, k_mov = theta
  
    # We can ignore normalization factor since it is constant.
    # So we simply return 0 for parameters in the specified range.
    if 0 <= k_div <= 1 and 0 <= k_bg <= 1 and 0 <= k_br <= 1 and 0 <= k_mov <= 1:
        return 0.0
    return -np.inf
    
def log_likelihood_coupled(theta, data, mu_n, sigma_n):
    '''
    Returns log of likelihood function for model with diffusion.
    
    Arguments:
        theta: model parameters (specified as a list)
                theta = [k_div, k_bg, k_br, k_mov]
        data: CellDen class object
        mu_n: mean of the log distribution of counting error
        sigma_n: standard deviation of the log distribution of counting error    
    '''
    model = solver_coupled(theta, data)[:,1:-1,1:-1,:]
    data_matrix = data.pd2np()[:,1:-1,1:-1,:]
    
    # Remove the bins in whcih observed number of cells is zero
    nonzero_args = np.nonzero(data_matrix)
    data_matrix = data_matrix[nonzero_args]
    model = model[nonzero_args]
        
    residual = (np.log(data_matrix) - np.log(model) - mu_n)**2
    chi_square = np.sum(residual/(sigma_n**2))
    constant = np.sum(np.log(1/np.sqrt(2.0*np.pi*sigma_n**2)))*residual.size
    logsum_data = np.sum(np.log(data_matrix))
                     
    return constant - logsum_data - 0.5*chi_square

def log_posterior_coupled(theta, data, mu_n, sigma_n):
    '''
    Returns log of posterior function for model with diffusion.
    
    Arguments:
        theta: model parameters (specified as a list)
                theta = [k_div, k_bg, k_br, k_mov]
        data: CellDen class object
        mu_n: mean of the log distribution of counting error
        sigma_n: standard deviation of the log distribution of counting error
    '''
    # If prior is -np.inf, no need to proceed so ends by returning -np.inf
    lp = log_prior_coupled(theta)
    if not np.isfinite(lp):
        return -np.inf
    
    return lp + log_likelihood_coupled(theta, data, mu_n, sigma_n)

def negative_log_posterior_coupled(theta, data, mu_n, sigma_n):
    '''
    Returns negative value of log_posterior_coupled function.
    
    Arguments:
        theta: model parameters (specified as a list)
                theta = [k_div, k_bg, k_br, k_mov]
        data: CellDen class object
        mu_n: mean of the log distribution of counting error
        sigma_n: standard deviation of the log distribution of counting error
    '''
    return -log_posterior_coupled(theta, data, mu_n, sigma_n)

def residual(theta, data, mu_n, sigma_n, coupled):
    
    '''
    Returns the residual from the likelihood function.
    Note that it is not same as the usual residual from the linear regression
    as the likelihood function depends on log-normal distribution.
    
    Argument:
        theta: model parameters (specified as a list)
               If coupled == True, theta = [k_div, k_bg, k_br, k_mov]
               If coupled == False, theta = [k_div, k_bg, k_br]
        data: CellDen class object
        mu_n: mean of the log distribution of counting error
        sigma_n: standard deviation of the log distribution of counting error
        coupled: True if one uses model with diffusion &
                Fals if one uses model without diffusion
    
    '''
    
    if coupled:
        y = data.pd2np()[:,1:-1,1:-1,:]
        m = solver_coupled(theta, data)[:,1:-1,1:-1,:]

        # Remove the bins in whcih observed number of cells is zero
        nonzero_args = np.nonzero(y)
        y = y[nonzero_args]
        m = m[nonzero_args]
    else:
        y = data.pd2np()
        m = solver_uncoupled(theta, data)

        # Remove the bins in whcih observed number of cells is zero
        nonzero_args = np.nonzero(y)
        y = y[nonzero_args]
        m = m[nonzero_args]

        # Remove the bins in whcih number of cells of the model is zero
        # These zero points become problems as we take log of them.
        # While it cannot be rigorously justified, the number of such case is
        # approximately less than 5%, so we expect contribution of these points
        # is negligible
        nonzero_args = np.nonzero(m)
        y = y[nonzero_args]
        m = m[nonzero_args]
        
    return (np.log(y) - np.log(m) - mu_n)/sigma_n

def run_mcmc(data, init_params, coupled, nwalkers=20, nsteps=500, spread=None,
             mu_n=-0.15, sigma_n=0.1, threadsNum=4):
    '''
    Runs MCMC using  Affine Invariant Markov chain Monte Carlo (MCMC) Ensemble 
    sampler from emcee package.
    Returns the sampler object after MCMC run.
    
    Arguments:
        data: CellDen class object
        init_params: Initial position to start MCMC
        coupled: True if one uses model with diffusion &
                Fals if one uses model without diffusion
        nwalkers: number of walkers for MCMC (optional)
        nsteps: number of steps for MCMC (optional)
        spread: parameter that spreads the starting position for MCMC (optional)
                It will be multiplied to the n-dimensional Gausssian ball to
                vary the starting position for each walker
        mu_n: mean of the log distribution of counting error (optional)
        sigma_n: standard deviation of the log distribution of counting error (optional)
        threadsNum: number of threads to be used for MCMC run (optional)
    '''
    if coupled:
        ndim = 4
    else:
        ndim = 3
        
    if spread==None:
        spread = 10**(np.floor(np.log10(init_params))-4)
    
    # Starting positions in Gaussian ball
    starting_positions = [init_params + spread*np.random.randn(ndim) \
                          for i in range(nwalkers)]
    

    # Set up the sampler object
    if coupled:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_coupled, 
                                    args=(data, mu_n, sigma_n), threads=threadsNum)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_uncoupled, 
                                    args=(data, mu_n, sigma_n), threads=threadsNum)    
    # Run the sampler.
    sampler.run_mcmc(starting_positions, nsteps)
    
    return sampler    

def plotMap(grid, duration, plotNum=10):
    '''
    Plot the heatmap of each type of cell over time
    
    Arguments:
        grid: diffeq solution arrays (3, binDiv, binDiv, time)
                            i.e. ([red,grn,both] x (binNum x binNum) x time)
        duration :  total timesteps
        plotNum : number of snapshots (plots). 
                  The time difference between each snapshot is even.    
    '''
    # Define time step
    time_step = int(duration/plotNum)
    grn = grid[0,:,:]
    red = grid[1,:,:]
    both = grid[2,:,:]
    # Plot heatmap for each time i*time_step
    for i in range(plotNum):
        plt.subplot(plotNum,3,1+i*3)
        sns.heatmap(both[:,:,i*time_step], vmin=0, vmax=150,
                    annot=True, fmt='.1f', cmap="Oranges")
        
        plt.subplot(plotNum,3,2+i*3)
        sns.heatmap(red[:,:,i*time_step], vmin=0, vmax=150,
                    annot=True, fmt='.1f', cmap="Reds")
    
        plt.subplot(plotNum,3,3+i*3)
        sns.heatmap(grn[:,:,i*time_step], vmin=0, vmax=150,
                    annot=True, fmt='.1f', cmap="Greens")
#    plt.tight_layout()
