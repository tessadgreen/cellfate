import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from cellfate import io

#import the .csv data file for cell location as dataframe
def load_data(data_file, CellA, CellB):

    both_high='Classify_Intensity_UpperQuartileIntensity_'+CellA+'_high_Intensity_UpperQuartileIntensity_'+CellB+'_high'
    both_low='Classify_Intensity_UpperQuartileIntensity_'+CellA+'_low_Intensity_UpperQuartileIntensity_'+CellB+'_low'
    high_CellA='Classify_Intensity_UpperQuartileIntensity_'+CellA+'_high_Intensity_UpperQuartileIntensity_'+CellB+'_low'
    high_CellB='Classify_Intensity_UpperQuartileIntensity_'+CellA+'_low_Intensity_UpperQuartileIntensity_'+CellB+'_high'

    data_loc=pd.read_csv(io.get_data_file_path(data_file), usecols=['ImageNumber',\
        both_high,both_low,high_CellA,high_CellB,'Location_Center_Y','Location_Center_X'])
    
    return (data_loc,both_high,both_low,high_CellA,high_CellB)

#Find the density of different types of cells at different times and bins

def cell_density(data_file, CellA, CellB, BinDiv, ImgWidth):
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
    ImgWidth: the width dimension of the image in pixels (e.g. for an image of 1024x1024, just enter 1024)
    
    return:
    -----------
        A dataframe of cell density, whose
            1) Main Columns are CellA, CellB and Both-Cell
            2) Sub Columns are different bins
            3) Rows are different time t
    '''
    data_path=io.get_data_file_path(data_file)
    data_loc,both_high,both_low,high_CellA,high_CellB=load_data(data_path,CellA,CellB)
    
    def bin_cell_den_at_one_t(t):
        Both_X=data_loc.loc[((data_loc['ImageNumber']==t)&((data_loc[both_high]==1)|(data_loc[both_low]==1))),\
                            'Location_Center_X'].values
        Both_Y=data_loc.loc[((data_loc['ImageNumber']==t)&((data_loc[both_high]==1)|(data_loc[both_low]==1))),\
                            'Location_Center_Y'].values
        CellA_X=data_loc.loc[((data_loc['ImageNumber']==t)&(data_loc[high_CellA]==1)),'Location_Center_X'].values
        CellA_Y=data_loc.loc[((data_loc['ImageNumber']==t)&(data_loc[high_CellA]==1)),'Location_Center_Y'].values
        CellB_X=data_loc.loc[((data_loc['ImageNumber']==t)&(data_loc[high_CellB]==1)),'Location_Center_X'].values
        CellB_Y=data_loc.loc[((data_loc['ImageNumber']==t)&(data_loc[high_CellB]==1)),'Location_Center_Y'].values
    
        BinWidth=np.floor_divide(ImgWidth,BinDiv)
        def one_bin_den(i,j):#bins are aranged in ith row and jth col
            # bin_index=i*BinDiv+j
            
            i=(BinDiv-1)-i
            BinY_Low=BinWidth*i-1
            BinY_High=BinWidth*(i+1)
            BinX_Low=BinWidth*j-1
            BinX_High=BinWidth*(j+1)
    
            BinArea=1 #arbitrary unit
            Both_Bin_Den=len(Both_X[(Both_X>BinX_Low)*(Both_X<BinX_High)*(Both_Y>BinY_Low)*(Both_Y<BinY_High)])/BinArea
            CellA_Bin_Den=len(CellA_X[(CellA_X>BinX_Low)*(CellA_X<BinX_High)*(CellA_Y>BinY_Low)*(CellA_Y<BinY_High)])/BinArea
            CellB_Bin_Den=len(CellB_X[(CellB_X>BinX_Low)*(CellB_X<BinX_High)*(CellB_Y>BinY_Low)*(CellB_Y<BinY_High)])/BinArea
            return [CellA_Bin_Den,CellB_Bin_Den,Both_Bin_Den]
        all_bin_den=np.vectorize(one_bin_den,otypes=[np.ndarray])
    
        bin_j,bin_i=np.meshgrid(np.arange(BinDiv),np.arange(BinDiv))
        cell_den_at_t=np.array(list(all_bin_den(bin_i,bin_j).flatten()))
    
        CellA_den_at_t=(cell_den_at_t[:,0]).flatten()
        CellB_den_at_t=(cell_den_at_t[:,1]).flatten()
        Both_den_at_t=(cell_den_at_t[:,2]).flatten()
    
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

# define a function to plot the locations of both-cells
def draw_cell_loc(data_file,CellA, CellB, time, BinDiv=1, bin_i=0, bin_j=0, ImgWidth=1024, colorBoth=[255/255,174/255,66/255],colorA='g',colorB='r'):
    '''
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
    
    time: from 0 to max time
    
    BinDiv: An integer telling the function to divide the orginal cell image into BinDiv x BinDiv bins
    bin_i, bin_j: from 0 to (maxmimum bin num-1), extracting the bin_i th row and the bin_j th column
         
    ImgWidth: the width dimension of the image in pixels (e.g. for an image of 1024x1024, just enter 1024
    
    '''
    t=time+1
    
    data_path=io.get_data_file_path(data_file)
    data_loc,both_high,both_low,high_CellA,high_CellB=load_data(data_path,CellA,CellB)
    
    #read the concerned time t
    Both_X=data_loc.loc[((data_loc['ImageNumber']==t)&((data_loc[both_high]==1)|(data_loc[both_low]==1))),\
                            'Location_Center_X'].values
    Both_Y=data_loc.loc[((data_loc['ImageNumber']==t)&((data_loc[both_high]==1)|(data_loc[both_low]==1))),\
                            'Location_Center_Y'].values
 
    CellA_X=data_loc.loc[((data_loc['ImageNumber']==t)&(data_loc[high_CellA]==1)),'Location_Center_X'].values
    CellA_Y=data_loc.loc[((data_loc['ImageNumber']==t)&(data_loc[high_CellA]==1)),'Location_Center_Y'].values
    CellB_X=data_loc.loc[((data_loc['ImageNumber']==t)&(data_loc[high_CellB]==1)),'Location_Center_X'].values
    CellB_Y=data_loc.loc[((data_loc['ImageNumber']==t)&(data_loc[high_CellB]==1)),'Location_Center_Y'].values
    
    #Find the coor in the specified bin
    BinWidth=np.floor_divide(ImgWidth,BinDiv)
    
    i=(BinDiv-1)-bin_i;j=bin_j
    
    BinY_Low=BinWidth*i-1
    BinY_High=BinWidth*(i+1)
    BinX_Low=BinWidth*j-1
    BinX_High=BinWidth*(j+1)
    
    concerned_Both_X=Both_X[(Both_X>BinX_Low)*(Both_X<BinX_High)*(Both_Y>BinY_Low)*(Both_Y<BinY_High)]
    concerned_Both_Y=Both_Y[(Both_X>BinX_Low)*(Both_X<BinX_High)*(Both_Y>BinY_Low)*(Both_Y<BinY_High)]
    concerned_CellA_X=CellA_X[(CellA_X>BinX_Low)*(CellA_X<BinX_High)*(CellA_Y>BinY_Low)*(CellA_Y<BinY_High)]
    concerned_CellA_Y=CellA_Y[(CellA_X>BinX_Low)*(CellA_X<BinX_High)*(CellA_Y>BinY_Low)*(CellA_Y<BinY_High)]
    concerned_CellB_X=CellB_X[(CellB_X>BinX_Low)*(CellB_X<BinX_High)*(CellB_Y>BinY_Low)*(CellB_Y<BinY_High)]
    concerned_CellB_Y=CellB_Y[(CellB_X>BinX_Low)*(CellB_X<BinX_High)*(CellB_Y>BinY_Low)*(CellB_Y<BinY_High)]
    
    #plot out the cell distribution
    plt.figure(figsize=(12,3))
    
    if BinDiv==1:
        title_end=' at time '+str(time)
    else:
        title_end=' in Bin '+str(bin_i*BinDiv+bin_j)+' at time '+str(time)
    
    plt.subplot(1,3,1,aspect='equal')
    plt.scatter(concerned_CellA_X,concerned_CellA_Y,color=colorA, s=1.5)
    if BinDiv==1:
        plt.xlim(0,ImgWidth);plt.ylim(ImgWidth,0)
    else:
        plt.xlim(BinX_Low+1,BinX_High);plt.ylim(BinY_High,BinY_Low+1)
    plt.title('Distribution of '+CellA+title_end)
    
    plt.subplot(1,3,2,aspect='equal')
    plt.scatter(concerned_CellB_X,concerned_CellB_Y,color=colorB, s=1.5)
    if BinDiv==1:
        plt.xlim(0,ImgWidth);plt.ylim(ImgWidth,0)
    else:
        plt.xlim(BinX_Low+1,BinX_High);plt.ylim(BinY_High,BinY_Low+1)
    plt.title('Distribution of '+CellB+title_end)
    
    plt.subplot(1,3,3,aspect='equal')
    plt.scatter(concerned_Both_X,concerned_Both_Y,color=colorBoth, s=1.5)
    if BinDiv==1:
        plt.xlim(0,ImgWidth);plt.ylim(ImgWidth,0)
    else:
        plt.xlim(BinX_Low+1,BinX_High);plt.ylim(BinY_High,BinY_Low+1)
    plt.title('Distribution of '+'Both-Cell'+title_end)
    
    plt.tight_layout()
    return

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
