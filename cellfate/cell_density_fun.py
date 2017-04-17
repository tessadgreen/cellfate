import numpy as np
import pandas as pd

# define a function to find the locations of both-cells
def loc_both(AX,AY,BX,BY,CellWidth):
    '''
    AX: x-coor of locations of cell type A
    AY: y-coor of locations of cell type A
    BX: x-coor of locations of cell type B
    BY: y-coor of locations of cell type B
    CellWidth: the width of the cell, in unit of pixels, suggested value=5
    
    '''
    def dist(x1,y1,x2,y2):
        return np.sqrt((x2-x1)**2+(y2-y1)**2)
    dist_1pt_to_otherpt=np.vectorize(dist,excluded=['x1','y1'])

    def min_dist(x1,y1):
        
        #consdier only 'close' points to save computational time
        prelim_select_close_pt=(((np.abs(BX-x1)<(CellWidth))*(np.abs(BY-y1)<(CellWidth)))==1) 
        
        if np.sum(prelim_select_close_pt)>0 : #if there is at least one 'close' point
            concern_BX=BX[prelim_select_close_pt]
            concern_BY=BY[prelim_select_close_pt]
            return np.min(dist_1pt_to_otherpt(x1,y1,concern_BX,concern_BY))
        else:
            #if no 'close' points
            return np.inf
    min_dist_vec=np.vectorize(min_dist)
    
    tmp_dist=min_dist_vec(AX,AY)
    BothX=AX[tmp_dist<(CellWidth/2)]
    BothY=AY[tmp_dist<(CellWidth/2)]
    return BothX,BothY

#divide into bins and calculate the density
def bins_density(LocX,LocY,BinDiv,ImgWidth): 
    '''
    This function will compute the density of that type of cell
    in each bin    
    
    LocX, LocY: location of cell of certain color/type with
        LocX, LocY be the x- and y- coordinates respectively.
    
    BinDiv: The original image will be divided into BinDiv x BinDiv bins.
    
    ImgWidth: the legnth of the original cell image, in pixels
    
    '''
    BinWidth=np.floor_divide(ImgWidth,BinDiv)
    def one_bin_density(i,j): #bins are aranged in ith row and jth col
        # bin_index=i*BinDiv+j
        BinX=LocX[((LocX>(BinWidth*i-1))*(LocX<(BinWidth*(i+1)+1))\
                   *(LocY>(BinWidth*j-1))*(LocY<(BinWidth*(j+1)+1)))]
        BinY=LocY[((LocX>(BinWidth*i-1))*(LocX<(BinWidth*(i+1)+1))\
                   *(LocY>(BinWidth*j-1))*(LocY<(BinWidth*(j+1)+1)))]
        BinArea=1 #arbitrary unit
        Den=len(BinX)/BinArea
        return Den
    all_bin_density=np.vectorize(one_bin_density,otypes=[np.ndarray])
    
    bin_j,bin_i=np.meshgrid(np.arange(BinDiv),np.arange(BinDiv))
    return np.array(list(zip(*all_bin_density(bin_i,bin_j)))[::-1])

#Find the density of different types of cells at different times and bins

#Find the density of different types of cells at different times and bins

def cell_density(CelltypeA,CelltypeB,data,CellWidth,BinDiv): 
    '''
    This function divides the original cell image into BinDiv x BinDiv bins 
    and calculates the density of different types of cell in each bin at 
    different times.
    
    Parameters:
    -----------
    CelltypeA: Name of the first cell type, dtype=string, e.g. 'Oct'
    CelltypeB: Name of the second cell type, dtype=string, e.g. 'Sox'
    data: A dictionary with index 
            'CelltypeWidth', containing the legnth size of the original cell image, 
                e.g. 'OctWidth'
            'CelltypeX', containing the x-coor of locations of the cell type, 
                e.g. 'OctX'
            'CelltypeY', containing the y-coor of locations of the cell type, 
                e.g. 'OctY'
        for both the first and second cell types.
    CellWidth: the width of the cell, in unit of pixels, suggested value=5
    BinDiv: An integer telling the function to divide the orginal cell image 
        into BinDiv x BinDiv bins 
    
    Returns:
    --------
    A dataframe with the density of different types of cells in the bins of the 
    original cell image at different time t.
    
    The format of the output dataframe would be like --
    Rows: time index
    Main Column: cell types
    Sub Column: bin index
    
    To extract the density of certain cell type at certain time of the bin bin_i,
    you can use: Name_of_Dataframe['CellName'][time][bin_i]
    e.g. density['Oct'][0:4][100] where density is the name of the dataframe, 
    'Oct' is the type of cell that is concerned, 0:4 is the times that we are 
    interested in, 100 is the bin that we are looking at
    
    '''
    #for CelltypeA
    WidthA_all=data[CelltypeA+'Width'][0]
    LocAX_all=data[CelltypeA+'X'][0]
    LocAY_all=data[CelltypeA+'Y'][0]
    
    tot_time=np.size(LocAX_all)
    ImgWidth=WidthA_all[0].flatten()[0]
    
    #for CelltypeB
    WidthB_all=data[CelltypeB+'Width'][0]
    LocBX_all=data[CelltypeB+'X'][0]
    LocBY_all=data[CelltypeB+'Y'][0]
    
    def cell_density_at_one_t(t):
        
        LocAX=LocAX_all[t].flatten()
        LocAY=LocAY_all[t].flatten()
        
        LocBX=LocBX_all[t].flatten()
        LocBY=LocBY_all[t].flatten()
        
        LocBothX,LocBothY=loc_both(LocAX,LocAY,LocBX,LocBY,CellWidth)
        DenA=bins_density(LocAX,LocAY,BinDiv,ImgWidth)
        DenB=bins_density(LocBX,LocBY,BinDiv,ImgWidth)
        DenBoth=bins_density(LocBothX,LocBothY,BinDiv,ImgWidth)
        return [DenA,DenB,DenBoth]
    cell_density_all_t=np.vectorize(cell_density_at_one_t,otypes=[np.ndarray])
    
    tmp_density=cell_density_all_t(np.arange(tot_time))
    
    #reorganize the calculated cell density into a dataframe
    density_A=np.zeros((tot_time,BinDiv*BinDiv))
    density_B=np.zeros((tot_time,BinDiv*BinDiv))
    density_Both=np.zeros((tot_time,BinDiv*BinDiv))
    for t in range(tot_time): 
        density_A[t,:]=tmp_density[t][0].flatten()
        density_B[t,:]=tmp_density[t][1].flatten()
        density_Both[t,:]=tmp_density[t][2].flatten()

    cols=pd.MultiIndex.from_tuples([ (x,y) \
    for x in [CelltypeA,CelltypeB,'Both'] for y in np.arange(BinDiv*BinDiv)])
    return pd.DataFrame(np.hstack((np.hstack((density_A,density_B)),density_Both)),\
    columns=cols)