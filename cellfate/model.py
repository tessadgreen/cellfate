# Numerical solver of the differential equation
# Reference: http://scipy-cookbook.readthedocs.io/items/CoupledSpringMassSystem.html

import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def vectorfield(w, t, p):
    """
    Defines the differential equations for the coupled spring-mass system.

    Arguments:
        w :  vector of the state variables:
                  w = [b,r,g]
        t :  time
        p :  vector of the parameters:
                  p = [k_ent,k_div,k_dep,k_bg,k_br]
    """
    b, r, g = w
    k_ent, k_div, k_dep, k_bg, k_br = p

    # Create f = (b',r',g'):
    f = [k_ent*b + (k_div - k_dep)*b - (k_bg+k_br)*b,
         k_ent*r + (k_div - k_dep)*r + k_br*b,
         k_ent*g + (k_div - k_dep)*g + k_bg*b,]
    return f

# Parameter values
k_ent = 0
k_div = 0.05
k_dep = 0
k_bg = 0.6
k_br = 0.1

# Bin number in each side
binNum = 4
# Initial conditions
# Grid of binNum x binNum where each position contains w0 = [b, r, g] 
init_grid = np.random.rand(binNum,binNum,3)*10

# ODE solver parameters
stoptime = 20.0
numpoints = 250

# Create the time samples for the output of the ODE solver.
t = np.linspace(0, stoptime, numpoints)

# Pack up the parameters and initial conditions:
p = [k_ent, k_div, k_dep, k_bg, k_br] 

# Create a grid to save the solved results
final_grid = np.zeros((binNum, binNum, 3, len(t)))

# Solve ODE for each bin
for i in range(binNum):
    for j in range(binNum):
        wsol = odeint(vectorfield, init_grid[i,j,:], t, args=(p,))
        final_grid[i,j,:,:] = wsol.T

def plotMap(grid, duration, plotNum=10):
    '''
    Plot the heatmap of each type of cell over time
    
    Arguments:
        grid: diffeq solution arrays (bins x [b,r,g] x time)
        duration :  time
        plotNum : number of snapshots (plots). 
                  The time difference between each snapshot is same.    
    '''
    # Define time step
    time_step = int(duration/plotNum)
    # Plot heatmap for each time i
    for i in range(plotNum):
        plt.subplot(plotNum,3,1+i*3)
        ax = sns.heatmap(grid[:,:,0,i*time_step], vmin=0, vmax=30, 
                         annot=True, fmt='.1f', cmap="Blues")
        
        plt.subplot(plotNum,3,2+i*3)
        ax = sns.heatmap(grid[:,:,1,i*time_step], vmin=0, vmax=30, 
                         annot=True, fmt='.1f', cmap="Reds")
    
        plt.subplot(plotNum,3,3+i*3)
        ax = sns.heatmap(grid[:,:,2,i*time_step], vmin=0, vmax=30, 
                         annot=True, fmt='.1f', cmap="Greens")


plt.figure()
# Plot the number of each cell over time in each bin        
for i in range(binNum):
    for j in range(binNum):
        plt.subplot(4, 4, 1+i*4+j)
        plt.xlabel('t')
        plt.ylabel('n')
        lw = 1
        b = final_grid[i,j,0,:]
        r = final_grid[i,j,1,:]
        g = final_grid[i,j,2,:]
        
        plt.plot(t, b, 'b', linewidth=lw)
        plt.plot(t, g, 'g', linewidth=lw)
        plt.plot(t, r, 'r', linewidth=lw)


plt.figure()
plotMap(final_grid, len(t), 5)
