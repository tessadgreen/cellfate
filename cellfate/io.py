import os
import numpy as np
import scipy.io as sio
import cell_density_fun
import cell_density_object

def get_data_file_path(filename, data_dir='test'):

    start = os.path.abspath('__file__')
    start_dir = os.path.dirname(start)

    data_dir = os.path.join(start_dir, data_dir)
    return os.path.join(start_dir, data_dir, filename)
    
def read(data_name, CelltypeA, CelltypeB, CellWidth, BinDiv):
    '''
    Parameters:
    -----------
    data_name: the name of the .mat metadata file of two cell types,
        which conatins index of
        'CelltypeWidth', containing the legnth of the original cell image, 
            e.g. 'OctWidth' or 'SoxWidth'
        'CelltypeX', containing the x-coor of locations of the cell type, 
            e.g. 'OctX' or 'SoxX'
        'CelltypeY', containing the y-coor of locations of the cell type, 
            e.g. 'OctY' or 'SoxY'
        for both the first and second cell types.
    CelltypeA: Name of the first cell type, dtype=string, e.g. 'Oct'
    CelltypeB: Name of the second cell type, dtype=string, e.g. 'Sox'
    CellWidth: the width of the nucleus, in unit of pixels, suggested value=5
    BinDiv: the original cell image will be divided into BinDiv x BinDiv bins,
        in which the cell density would be intended to be calulated
        
    Return:
    -----------
    A class object containing
    data: the density of different types of cells in different bins in a dataframe
    cellwidth: the length of the nucleus
    bin_num: total numbers of bins
    tot_time: total time steps of the data
    
    '''
    data_path=get_data_file_path(data_name)
    data_raw=sio.loadmat(data_path)
    data=cell_density_fun.cell_density(CelltypeA,CelltypeB,data_raw,CellWidth,BinDiv)
    return cell_density_object.CellDen(data, CellWidth)

