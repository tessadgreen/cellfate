import os
import numpy as np
import scipy.io as sio

def get_data_file_path(filename, data_dir='data'):

    start = os.path.abspath(__file__)
    start_dir = os.path.dirname(start)

    data_dir = os.path.join(start_dir, data_dir)
    return os.path.join(start_dir, data_dir, filename)

def read(data_file, CellWidth, BinDiv):
    return CellLoc(data_file, CellWidth, BinDiv)

class CellLoc:
    """
    data_file: the path to the .mat metadata file of two cell types,
        which conatins index of
        'CelltypeWidth', containing the legnth of the original cell image, 
            e.g. 'OctWidth' or 'SoxWidth'
        'CelltypeX', containing the x-coor of locations of the cell type, 
            e.g. 'OctX' or 'SoxX'
        'CelltypeY', containing the y-coor of locations of the cell type, 
            e.g. 'OctY' or 'SoxY'
        for both the first and second cell types.
    
    CellWidth: the width of the cell, in unit of pixels, suggested value=5
    
    BinDiv: the original cell image will be divided into BinDiv x BinDiv bins,
        in which the cell density would be intended to be calulated
        
    """
    
    def __init__(self, data_file, CellWidth, BinDiv):
        self.data_file = data_file
        self.CellWidth = CellWidth
        self.BinDiv = BinDiv
        
        all_data =  sio.loadmat(data_file)
        self.data = all_data #store in a dictionary for convenience 
