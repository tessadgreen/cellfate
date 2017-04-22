# cellfate

Fitting parameters of the model for density of human embryonic stem cells in processed images at different times t to differentiate the fates of the cells.

## Files:

- model.ipynb: a jupyter notebook showing the mathematical details of the differential equation used to model the denisty of different cell types at different time t.

- cell_density_fun.py: contains the functions to calculate densities of different types of cells in different bins at different times.

- cell_density_object.py: contains a class to put the follwing items into an object 
    1) the dataframe of the process cell density data 
    2) information of the size of a nulcues, num of bins and total time step of the data

- io.py: contains the functions to load the metadata from a .mat file with index 
    'CelltypeWidth', containing the legnth of the original cell image, (e.g. 'OctWidth' or 'SoxWidth')
    'CelltypeX', containing the x-coor of locations of the cell type, (e.g. 'OctX' or 'SoxX')
    'CelltypeY', containing the y-coor of locations of the cell type, (e.g. 'OctY' or 'SoxY')
  for both the first and second cell types and it will output an object containing the data of cell density descibed above in 'cell_density_object.py'.

- model.py: contains the functions to solve ODE for our model and to calculate likelihood function

- test_data.npz: a sample dataset to create the following two simulated data
    1) sample.pkl: simulated data (2x2 bins) in the form of pandas dataframe. The parameters used to create this dataset is [k_div, k_bg, k_br]=[0.05, 0.6, 0.1]
    2) sample_4x4.pkl: simulated data (4x4 bins) in the form of pandas dataframe. The parameters used to create this dataset is [k_div, k_bg, k_br]=[0.1, 0.4, 0.2]
    3) sample_2x2_30.pkl: simulated data (2x2 bins) in the form of pandas dataframe for unit timestep of 30. The parameters used to create this dataset is [k_div, k_bg, k_br]=[0.1, 0.4, 0.2]  

## License:

GNU General Public License v3

## Authors:

- Tessa Green
- Yau Chuen (Oliver) Yam
- Seung Hwan Lee
