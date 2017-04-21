import os
import scipy.io as sio
from cellfate import cell_density_fun, cell_density_object
import pandas as pd

def get_data_file_path(filename, data_dir='test'):

    start = os.path.abspath(__file__)
    start_dir = os.path.dirname(start)

    data_dir = os.path.join(start_dir,data_dir)
    return os.path.join(start_dir,data_dir,filename)


def read_csv(data_name, CelltypeA='Sox2', CelltypeB='Oct4', BinDiv, data_dir='test'):
    '''
    Parameters:
    -----------
    data_name: the name of the .csv file containing the data with the following columns
        Identity Labeling:
            Column 0:'ImageNumber': denoting the time step image
            Column 1: 'ObjectNumber': denoting the arbitrary identity of the cell
        Intensity classifications:
            'Classify_Intensity_UpperQuartileIntensity_Sox2_high_Intensity_UpperQuartileIntensity_Oct4_high'
            'Classify_Intensity_UpperQuartileIntensity_Sox2_high_Intensity_UpperQuartileIntensity_Oct4_low'
            'Classify_Intensity_UpperQuartileIntensity_Sox2_low_Intensity_UpperQuartileIntensity_Oct4_high'
            'Classify_Intensity_UpperQuartileIntensity_Sox2_low_Intensity_UpperQuartileIntensity_Oct4_low'
        Locations:
            'Location_Center_X'
            'Location_Center_Y'

    return:
        A class object containing 
            data, bin_num, tot_time
    '''
    #needs to be modified to take different protein names
    both_high='Classify_Intensity_UpperQuartileIntensity_Sox2_high_Intensity_UpperQuartileIntensity_Oct4_high'
    both_low='Classify_Intensity_UpperQuartileIntensity_Sox2_low_Intensity_UpperQuartileIntensity_Oct4_low'
    high_Oct4='Classify_Intensity_UpperQuartileIntensity_Sox2_low_Intensity_UpperQuartileIntensity_Oct4_high'
    high_Sox2='Classify_Intensity_UpperQuartileIntensity_Sox2_high_Intensity_UpperQuartileIntensity_Oct4_low'



    data_path=get_data_file_path(data_name)
    data_full=pd.read_csv(data_path, usecols=['ImageNumber',
        both_high,both_low,high_Sox2,high_Oct4,'Location_Center_Y','Location_Center_X'])

    data=cell_density_csv(data_full, CelltypeA, CelltypeB, BinDiv)
    return data

def cell_density_csv(input_data, CelltypeA, CelltypeB, BinDiv):
    '''
    This function divides the original data into (BinDiv, BinDiv) bins 
    and calculates the density of different types of cell in each bin at 
    different times.

    Parameters:
    -----------
    CelltypeA:
    CelltypeB:
    input_data: a pandas data frame as created by read_csv
    BinDiv: an integer telling the function how many sub-images to measure density in
    '''

    both_high='Classify_Intensity_UpperQuartileIntensity_Sox2_high_Intensity_UpperQuartileIntensity_Oct4_high'
    both_low='Classify_Intensity_UpperQuartileIntensity_Sox2_low_Intensity_UpperQuartileIntensity_Oct4_low'
    high_Oct4='Classify_Intensity_UpperQuartileIntensity_Sox2_low_Intensity_UpperQuartileIntensity_Oct4_high'
    high_Sox2='Classify_Intensity_UpperQuartileIntensity_Sox2_high_Intensity_UpperQuartileIntensity_Oct4_low'
    
    #should instead return data that's had it density processed as 
    #in cell_density_fun previously
    return input_data


def read(data_name, CelltypeA, CelltypeB, CellWidth, BinDiv):
    '''
    Parameters:
    -----------
    data_name: the name of the .mat metadata file of two cell types,
        which contains index of
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

