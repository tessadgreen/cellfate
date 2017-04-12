
'''data comes in as a matlab cell array
	you can open this in Matlab and view data using the cell2mat command


Attributes:
	Number_Object_Number
	Location_Center_X
	Location_Center_Y

handles.Measurements.OctNuclei.Number_Object_Number
handles.Measurements.SoxNuclei.Number_Object_Number
or similar


In this file we will:
1) import the matlab file so that python can access its contents

2) Use locations to identify cells that are "both"
		-probably set up some "nearness threshold" where when centers
		are sufficiently close, we class those as a 'both' cell

3) Divide the images into analysis bins, 
	calculating the n_b, n_g, n_r for each bin at each time

4) Put this data into an instance of our class:
	a pd.DataFrame with categories for time, n_b, n_r, n_g, bin

'''