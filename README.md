# cellfate

This package uses cellular location and time data for differentiating cells to estimate the differentiation rate for different cell times in vitro. Input data is in the form of the locations and states of nuclei for a time series of images. Bayesian inference is then used to fit the differentiation rates as well as the rate of cell division.
 
Differentiating embryonic stem cells make cell-fate decisions in vitro. These decisions determine what cell type--neuron, muscle, fibroblast, and many more--the cells would go on to become. These decisions are encoded by transcription factor activity in cells. Each transcription factor regulates many other genes, including other transcription factors. This allows for the diversity of cell types present in multicellular organisms. The first decision cells make, between mesendoderm and neurectoderm, can be seen in changing levels of two transcription factors, Oct4 and Sox2. Using CRISPR, these two factors have been fused to fluorescent reporters. As a result, these cells change color as they make their first fate decision. They start off in a "both" state, bright in both red and green. Then, some cells move to a "red" state, expressing only Oct4, and others move to a "green" state, expressing only Sox2. This package analyses time lapses of this process to infer the dynamics of this process.

Cells should have fluorescently tagged transcription factors that provide real-time reporting of cell state. Both reporters are active in the pre-decision state, and a different one is down-regulated in each differentiated state. Starting images are of live cells in vitro.

## Files:

- model.ipynb: a jupyter notebook showing the mathematical details of the differential equation used to model the denisty of different cell types at different time t.

- celldensity.py: contains 
    1) the function to calculate densities of different types of cells in different bins at different times.
    2) the function to plot the cell distribution at certain time and certain bin
    3) contains a class to put the follwing attribute ->
       * the dataframe of the process cell density data 
       * information of the names of the cell, num of bins, length scale, time scale and total time step of the data
       * a function to plot the heatmap of the cell density

- io.py: contains the functions to load the .csv file containing the data with the cell location information and it will output an object containing the data of cell density descibed in 'celldensity.py'.

- model.py: contains the functions to solve ODE for our model and to calculate likelihood function

- Simulated data
    1) sample.pkl: simulated data (2x2 bins) in the form of pandas dataframe. The parameters used to create this dataset is [k_div, k_bg, k_br]=[0.05, 0.6, 0.1]
    2) sample_4x4.pkl: simulated data (4x4 bins) in the form of pandas dataframe. The parameters used to create this dataset is [k_div, k_bg, k_br]=[0.1, 0.4, 0.2]
    3) sample_2x2_30.pkl: simulated data (2x2 bins) in the form of pandas dataframe for unit timestep of 30. The parameters used to create this dataset is [k_div, k_bg, k_br]=[0.1, 0.4, 0.2]  

## License:

GNU General Public License v3

## Authors:

- Tessa Green
- Yau Chuen (Oliver) Yam
- Seung Hwan Lee
