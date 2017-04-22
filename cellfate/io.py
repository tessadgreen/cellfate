import os
import scipy.io as sio
from cellfate import cell_density_fun, cell_density_object
import pandas as pd
import numpy as np

def get_data_file_path(filename, data_dir='test'):

    start = os.path.abspath(__file__)
    start_dir = os.path.dirname(start)

    data_dir = os.path.join(start_dir,data_dir)
    return os.path.join(start_dir,data_dir,filename)
    
def read(data_file, CellA, CellB, BinDiv):
    '''
    Parameters:
    -----------
    data_file: the .csv file containing the data with the following columns
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
            
    CellA: Name of the first cell type, e.g.'Sox2'
    CellB: Name of the second cell type, e.g. 'Oct4'
    BinDiv: An integer telling the function to divide the orginal cell image into BinDiv x BinDiv bins     
    
    return:
    -----------
        A dataframe of cell density, whose
            1) Main Columns are CellA, CellB and Both-Cell
            2) Sub Columns are different bins
            3) Rows are different time t
    '''
    
    
    both_high='Classify_Intensity_UpperQuartileIntensity_'+CellA+'_high_Intensity_UpperQuartileIntensity_'+CellB+'_high'
    both_low='Classify_Intensity_UpperQuartileIntensity_'+CellA+'_low_Intensity_UpperQuartileIntensity_'+CellB+'_low'
    high_CellA='Classify_Intensity_UpperQuartileIntensity_'+CellA+'_high_Intensity_UpperQuartileIntensity_'+CellB+'_low'
    high_CellB='Classify_Intensity_UpperQuartileIntensity_'+CellA+'_low_Intensity_UpperQuartileIntensity_'+CellB+'_high'

    data_loc=pd.read_csv(get_data_file_path(data_file), usecols=['ImageNumber',\
        both_high,both_low,high_CellA,high_CellB,'Location_Center_Y','Location_Center_X'])
    
    def bin_cell_den_at_one_t(t):
        Both_X=data_loc.loc[((data_loc['ImageNumber']==t)&((data_loc[both_high]==1)|(data_loc[both_low]==1))),\
                            'Location_Center_X'].values
        Both_Y=data_loc.loc[((data_loc['ImageNumber']==t)&((data_loc[both_high]==1)|(data_loc[both_low]==1))),\
                            'Location_Center_Y'].values
        CellA_X=data_loc.loc[((data_loc['ImageNumber']==t)&(data_loc[high_CellA]==1)),'Location_Center_X'].values
        CellA_Y=data_loc.loc[((data_loc['ImageNumber']==t)&(data_loc[high_CellA]==1)),'Location_Center_Y'].values
        CellB_X=data_loc.loc[((data_loc['ImageNumber']==t)&(data_loc[high_CellB]==1)),'Location_Center_X'].values
        CellB_Y=data_loc.loc[((data_loc['ImageNumber']==t)&(data_loc[high_CellB]==1)),'Location_Center_Y'].values
    
        ImgWidth=1024
        BinWidth=np.floor_divide(ImgWidth,BinDiv)
        def one_bin_den(i,j):#bins are aranged in ith row and jth col
            # bin_index=i*BinDiv+j
            BinX_Low=BinWidth*i-1
            BinX_High=BinWidth*(i+1)+1
            BinY_Low=BinWidth*j-1
            BinY_High=BinWidth*(j+1)+1
    
            BinArea=1 #arbitrary unit
            Both_Bin_Den=len(Both_X[(Both_X>BinX_Low)*(Both_X<BinX_High)*(Both_Y>BinY_Low)*(Both_Y<BinY_High)])/BinArea
            CellA_Bin_Den=len(CellA_X[(CellA_X>BinX_Low)*(CellA_X<BinX_High)*(CellA_Y>BinY_Low)*(CellA_Y<BinY_High)])/BinArea
            CellB_Bin_Den=len(CellB_X[(CellB_X>BinX_Low)*(CellB_X<BinX_High)*(CellB_Y>BinY_Low)*(CellB_Y<BinY_High)])/BinArea
            return [CellA_Bin_Den,CellB_Bin_Den,Both_Bin_Den]
        all_bin_den=np.vectorize(one_bin_den,otypes=[np.ndarray])
    
        bin_j,bin_i=np.meshgrid(np.arange(BinDiv),np.arange(BinDiv))
        cell_den_at_t=np.array(list(all_bin_den(bin_i,bin_j).flatten()))
    
        CellA_den_at_t=np.array(list(zip(*np.reshape(cell_den_at_t[:,0],(BinDiv,BinDiv))))[::-1]).flatten()
        CellB_den_at_t=np.array(list(zip(*np.reshape(cell_den_at_t[:,1],(BinDiv,BinDiv))))[::-1]).flatten()
        Both_den_at_t=np.array(list(zip(*np.reshape(cell_den_at_t[:,2],(BinDiv,BinDiv))))[::-1]).flatten()
    
        return [CellA_den_at_t,CellB_den_at_t,Both_den_at_t]
    bin_cell_den_at_all_t=np.vectorize(bin_cell_den_at_one_t,otypes=[np.ndarray])

    max_t=data_loc['ImageNumber'].max()
    cell_den_diff_t=bin_cell_den_at_all_t(np.arange(max_t)+1)
    
    CellA_den=np.zeros((max_t,BinDiv**2))
    CellB_den=np.zeros((max_t,BinDiv**2))
    Both_den=np.zeros((max_t,BinDiv**2))
    for t in range(max_t):
        CellA_den[t,:]=cell_den_diff_t[t][0]
        CellB_den[t,:]=cell_den_diff_t[t][1]
        Both_den[t,:]=cell_den_diff_t[t][2]
        
    cols=pd.MultiIndex.from_tuples([ (x,y) for x in [CellA,CellB,'Both'] for y in np.arange(BinDiv*BinDiv)])
    return pd.DataFrame(np.hstack((np.hstack((CellA_den,CellB_den)),Both_den)),columns=cols)

