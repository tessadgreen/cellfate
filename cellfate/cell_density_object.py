import os
import numpy as np
import scipy.io as sio

class CellDen:
    """
    data: the processed data for densities of types of cell in a dataframe
    
    CellWidth: the width of the nucleus, in unit of pixels, suggested value=5
        
    """
    
    def __init__(self, data, CellWidth):
        self.data = data
        self.cellwidth = CellWidth
        
        tot_time,bin_num_tmp=np.shape(data)
        bin_num=np.sqrt(bin_num_tmp/3)
        self.bin_num = bin_num
        self.tot_time = tot_time
