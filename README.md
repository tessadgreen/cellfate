# cellfate

Fitting parameters of the model for density of human embryonic stem cells in processed images at different times t to differentiate the fates of the cells.

## Files:

- model.ipynb: a jupyter notebook showing the mathematical details of the differential equation used to model the denisty of different cell types at different time t.

- io.py: contains the functions to load the metadata from a .mat file with index 
    'CelltypeWidth', containing the legnth of the original cell image, (e.g. 'OctWidth' or 'SoxWidth')
    'CelltypeX', containing the x-coor of locations of the cell type, (e.g. 'OctX' or 'SoxX')
    'CelltypeY', containing the y-coor of locations of the cell type, (e.g. 'OctY' or 'SoxY')
  for both the first and second cell types.
  
- cell_density_in_bins.py: contains the function to calculate densities of different types of cells in different bins at different times.

## License:

GNU General Public License v3

## Authors:

- Tessa Green
- Yau Chuen (Oliver) Yam
- Seung Hwan Lee
