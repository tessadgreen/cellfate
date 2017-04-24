import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class CellDen:
    """
    Parameters:
    -----------
    data: the processed data for densities of types of cell in a dataframe
    
    Attributes:
    -----------
    CellDen.data: A dataframe with the density of different types of cells in the bins of the 
    original cell image at different time t.The format of the output dataframe would be like --
        Rows-> time index
        Main Column-> cell types
        Sub Column-> bin index
    
    CellDen.cellname: a tuple of the first and second cell name
    CellDen.bin_num: total bin number
    CellDen.tot_time: total time duratiron of the experiment
    
    """
    
    def __init__(self, data):
        self.data = data
        
        CellA=list(data.columns.levels)[0][2]
        CellB=list(data.columns.levels)[0][1]
        self.cellname=(CellA,CellB)
        
        tot_time,bin_num_tmp=np.shape(data)
        bin_num=np.sqrt(bin_num_tmp/3)
        self.bin_num = int(bin_num)
        self.tot_time = tot_time

    def pd2np(self):
        '''
        Transforms pandas dataframe of input CellDen object into numpy array
        The resulting ndarray will have a form of (3, binDiv, binDiv, time):
        [red, green, both] x (binDiv x binDiv) x(time)
        
        Arguments:
            input_data: CellDen class object
        '''
        return np.reshape(self.data.as_matrix().T, (3,self.bin_num, self.bin_num, -1))
        
        
    def plotMap(self, plotNum=3):
        '''
        Plot the heatmap of each type of cell over time
        
        Arguments:
            grid: ndarray of (3, binDiv, binDiv, time)
                                i.e. ([red,grn,both] x (binNum x binNum) x time)
            duration :  total timesteps
            plotNum : number of snapshots (plots). 
                      The time difference between each snapshot is even.    
        '''
        grid = self.pd2np()
        # Define time step
        time_step = int(self.tot_time/plotNum)
        grn = grid[0,:,:]
        red = grid[1,:,:]
        both = grid[2,:,:]
        # Plot heatmap for each time i*time_step
        for i in range(plotNum):
            plt.subplot(plotNum,3,1+i*3)
            sns.heatmap(both[:,:,i*time_step], vmin=0, vmax=1000,
                        annot=True, fmt='.1f', cmap="Oranges")
            
            plt.subplot(plotNum,3,2+i*3)
            sns.heatmap(red[:,:,i*time_step], vmin=0, vmax=1000, 
                        annot=True, fmt='.1f', cmap="Reds")
        
            plt.subplot(plotNum,3,3+i*3)
            sns.heatmap(grn[:,:,i*time_step], vmin=0, vmax=1000, 
                        annot=True, fmt='.1f', cmap="Greens")