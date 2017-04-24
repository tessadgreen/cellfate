import numpy as np

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