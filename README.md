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
  for both the first and second cell types and it will output an object descibed above in 'cell_density_object.py'.

- model.py: contains the functions to solve ODE for our model and to calculate likelihood function

## License:

GNU General Public License v3

## Authors:

- Tessa Green
- Yau Chuen (Oliver) Yam
- Seung Hwan Lee
