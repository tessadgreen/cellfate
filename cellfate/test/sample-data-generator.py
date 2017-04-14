import numpy as np
from numpy import random
import pandas as pd 
import cellfate

"""
	The function gen_data can be used to generate fake processed 
	data with known parameter values for testing our model.
	This model treats time steps as discrete rather than continuous,
	so is a fairly crude simulation, but we should still be able 
	to recover comparable parameters.

"""


def sum_neighbors(matrix):
	"""
	Returns a matrix whose elements are 
	the average of the 4 adjacent entries in the original matrix
	"""
	(a,b)=np.shape(matrix)
	result=np.zeros((a,b))
	result[0,0]=(matrix[1,0]+matrix[0,1])*0.5
	result[a-1,b-1]=(matrix[a-2,b-1]+matrix[a-1,b-2])*0.5
	result[a-1,0]=(matrix[a-2,0]+matrix[a-1,1])*0.5
	result[0,b-1]=(matrix[0,b-2]+matrix[1,b-1])*0.5

	for x in range(1,a-1):
		result[x,0]=(matrix[x-1,0]+matrix[x+1,0]+matrix[x,1])/3
		result[x,b-1]=(matrix[x-1,b-1]+matrix[x+1,b-1]+matrix[x,b-2])/3


	for y in range(1,b-1):
		result[0,y]=(matrix[0,y-1]+matrix[0,y+1]+matrix[1,y])/3
		result[a-1,y]=(matrix[a-2,y]+matrix[a-1,y-1]+matrix[a-1,y+1])/3
		for x in range(1,a-1):
			result[x,y]=(matrix[x+1,y]+matrix[x-1,y]+matrix[x,y-1]+matrix[x,y+1])/4


	return result



def gen_data(k, N_bins,T_steps,sigma):
	"""
		Generates fake processed data for model testing
	"""
	#test set of model parameter k
	#k = [k_entry, k_delta=k_division-k_departure, k_bg, k_br, k_silence]
	[k_entry, k_delta, k_bg, k_br, k_silence]=k
	N_bins_x,N_bins_y=N_bins
	#starting bin occupancies


	n_b=np.zeros((N_bins_x,N_bins_y,T_steps))
	n_r=np.zeros((N_bins_x,N_bins_y,T_steps))
	n_g=np.zeros((N_bins_x,N_bins_y,T_steps))


	n_b[:,:,0]=np.random.randint(100,size=N_bins)
	n_r[:,:,0]=np.random.randint(10,size=N_bins)
	n_g[:,:,0]=np.random.randint(10,size=N_bins)

	#print(n_b[:,:,0])
	dn_b=np.zeros((N_bins_x,N_bins_y))
	dn_g=np.zeros((N_bins_x,N_bins_y))
	dn_r=np.zeros((N_bins_x,N_bins_y))


	for t in range(1,T_steps):
		#print(t)

		n_b_neighbors=sum_neighbors(n_b[:,:,t-1])
		n_g_neighbors=sum_neighbors(n_g[:,:,t-1])
		n_r_neighbors=sum_neighbors(n_r[:,:,t-1])
		#
		#if t==1: print(n_b_neighbors)

		for x in range(N_bins_x):
			for y in range(N_bins_y):
				dn_b[x,y]=(k_entry*n_b_neighbors[x,y]*4
					+ k_delta*n_b[x,y,t-1]
					-(k_bg+k_br)*n_b[x,y,t-1]
					-k_silence*n_b[x,y,t-1]
					)
				dn_g[x,y]=(k_entry*n_g_neighbors[x,y]*4
					+ k_delta*n_g[x,y,t-1]
					+ k_bg*n_b[x,y,t-1] )
				dn_r[x,y]=(k_entry*n_r_neighbors[x,y]*4
					+ k_delta*n_r[x,y,t-1]
					+ k_br*n_b[x,y,t])

		#if t==1: print(dn_b)

		n_b[:,:,t]=n_b[:,:,t-1]+dn_b+ random.normal(0,sigma[0],N_bins)+random.normal(0,sigma[1],N_bins)
		n_r[:,:,t]=n_r[:,:,t-1]+dn_r+ random.normal(0,sigma[0],N_bins)+random.normal(0,sigma[1],N_bins)
		n_g[:,:,t]=n_g[:,:,t-1]+dn_g+ random.normal(0,sigma[0],N_bins)+random.normal(0,sigma[1],N_bins)

		#if t==1: print(n_b[:,:,t])

	#print(dn_b)
	#increment using diffeq plus noise terms

	#generate random nuclei locations in bins 
	#processed_data=pd.DataFrame({'n_b': n_b, 'n_r': n_r, 'n_g': n_g, })

	return n_b,n_r,n_g


k=[0.02, 0.005, 0.014, 0.025, 0.05]
N_bins=(4,5)
T_steps=5
sigma=(.02,.05) #(sigma_obs, sigma_cells)

#sprint(sum_neighbors(np.array(np.ones((5,8)))))
(alpha,beta,gamma)=gen_data(k, N_bins, T_steps,sigma)

	
thing=DataSet(4,5)

#data=pd.DataFrame('x'=[]
#	'y'=
#	't')
#print(alpha[:,:,T_steps-1])
#print(beta[:,:,T_steps-1])
#DataSet

#print(np.shape(np.zeros((5,2))))