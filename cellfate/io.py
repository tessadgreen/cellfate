import os
from cellfate import celldensity
import numpy as np
import pandas as pd

def get_data_file_path(filename, data_dir='test'):

    start = os.path.abspath(__file__)
    start_dir = os.path.dirname(start)

    data_dir = os.path.join(start_dir,data_dir)
    return os.path.join(start_dir,data_dir,filename)

#import the .csv data file for cell location as dataframe
def load_data(data_file, CellA, CellB):
    """
    Import raw data from a csv

    data_file: filename
    CellA: Name of the first cell type, e.g.'Sox2'
    CellB: Name of the second cell type, e.g. 'Oct4'
    """

    both_high='Classify_Intensity_UpperQuartileIntensity_'+CellA+'_high_Intensity_UpperQuartileIntensity_'+CellB+'_high'
    both_low='Classify_Intensity_UpperQuartileIntensity_'+CellA+'_low_Intensity_UpperQuartileIntensity_'+CellB+'_low'
    high_CellA='Classify_Intensity_UpperQuartileIntensity_'+CellA+'_high_Intensity_UpperQuartileIntensity_'+CellB+'_low'
    high_CellB='Classify_Intensity_UpperQuartileIntensity_'+CellA+'_low_Intensity_UpperQuartileIntensity_'+CellB+'_high'

    data_loc=pd.read_csv(get_data_file_path(data_file), usecols=['ImageNumber',\
        both_high,both_low,high_CellA,high_CellB,'Location_Center_Y','Location_Center_X'])

    return (data_loc,both_high,both_low,high_CellA,high_CellB)

def read(data_file, CellA, CellB, BinDiv, ImgWidth=1024, time_scale=0.25, length_scale=1.33):
    '''
    Parameters:
    -----------
    data_file: the .csv file containing the data with the following columns
        Identity Labeling:
            Column 0:'ImageNumber': denoting the time step image
            Column 1: 'ObjectNumber': denoting the arbitrary identity of the cell
        Intensity classifications:
            'Classify_Intensity_UpperQuartileIntensity_CellA_high_Intensity_UpperQuartileIntensity_CellB_high'
            'Classify_Intensity_UpperQuartileIntensity_CellA_high_Intensity_UpperQuartileIntensity_CellB_low'
            'Classify_Intensity_UpperQuartileIntensity_CellA_low_Intensity_UpperQuartileIntensity_CellB_high'
            'Classify_Intensity_UpperQuartileIntensity_CellA_low_Intensity_UpperQuartileIntensity_CellB_low'
            where the order of CellA and CellB should be the same as that of the input parameter
        Locations:
            'Location_Center_X'
            'Location_Center_Y'

    CellA: Name of the first cell type, e.g.'Sox2'
    CellB: Name of the second cell type, e.g. 'Oct4'
    BinDiv: An integer telling the function to divide the orginal cell image into BinDiv x BinDiv bins
    ImgWidth: the width dimension of the image in pixels (e.g. for an image of 1024x1024, just enter 1024)
    time_scale: time scale of each time step between subsequent images, in hours
    length_scale: linear size scale of the image, in mm

    Returns:
    -----------
    An object with the following attributes:

    CellDen.data: A dataframe with the density of different types of cells in the bins of the
    original cell image at different time t.The format of the output dataframe would be like --
        Rows-> time index
        Main Column-> cell types
        Sub Column-> bin index

    CellDen.cellname: a tuple of the first and second cell name
    CellDen.bin_num: total bin number
    CellDen.tot_time: total time step of the experiment
    CellDen.time_scale: time scale between subsequent time steps, in hours
    CellDen.length_scale: linear size scale of the image, in mm
    '''
    data=celldensity.cell_density(data_file,CellA,CellB,BinDiv,ImgWidth)
    return celldensity.CellDen(data,time_scale,length_scale)
