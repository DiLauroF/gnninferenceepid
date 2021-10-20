#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 20:30:18 2021

@author: fra
"""

import numpy as np

'''
This model is for Italy. It has now 17 communities, whose sizes are informed
from real world data (See https://www.nature.com/articles/s41467-020-20544-y)

This is a more structured test, that should prove to be more difficult.
'''


#Italy
country = "Italy"
setting = "_country_level_M_overall_contact_matrix_85.csv"
#setting = "_country_level_F_school_setting_85.csv"
population = "_country_level_age_distribution_85.csv"

##Massachussets
# country='United_States_subnational_'
# setting='Massachusetts_M_overall_contact_matrix_85.csv'
# population = 'Massachusetts_age_distribution_85.csv'
# setting = "Massachusetts_F_school_setting_85.csv"

pattern = country+setting
contact_matrix=np.loadtxt(pattern,ndmin=2,delimiter=',')

agedist = np.loadtxt(country+population, delimiter =',')

def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""

    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))

#Symmetrize and check the diagonal
#contact_matrix_sym = (contact_matrix + contact_matrix.T)/4
#np.fill_diagonal(contact_matrix_sym,np.diag(contact_matrix))

#sanity check
'''
for i in range(85):
    for j in range(i):
        if (agedist[i,1]*contact_matrix[i,j] - agedist[j,1]*contact_matrix[j,i])>1:
            print(i,j)
'''    

newmat = np.zeros((85,85))
for i in range(85):
    newmat[i] = contact_matrix[i]*agedist[i,1]



matblocks = split(newmat,5,5)


#I am summing all the blocks to get the reduced matrix
contact_matrix = np.apply_over_axes(np.sum, matblocks, [1,2]).reshape((17,17))


newage = np.zeros(17)
for i in range(17):
    newage[i] = agedist[5*i:5*i+5,1].sum()
    contact_matrix[i] /= newage[i]

p = np.zeros((17,17))

percentages = newage/np.sum(newage)
N = 1000

for i in range(17):
    for j in range(i):
        p[i][j] = contact_matrix[i,j]/(percentages[i]*N)
        p[j][i] = p[i][j]


sizes = np.round(N*percentages).astype(int)

import networkx as nx
#p = contact_matrix/np.reshape(np.outer(np.sqrt(sizes),np.sqrt(sizes)),(17,17))

import pandas
graph_genaration = 80
for graph_no in range(graph_genaration):
    A = nx.stochastic_block_model(sizes,p)
    degrees = [A.degree(n) for n in A.nodes()]
    gamma_E = 1/4   #rate from E to I
    gamma_I = 1/7   #rate from I to R/D
    R_0 = 3
    tauf = 200     #final time
    pd = 0.07       #probability of dying
    i0 = 4          #Initially infected nodes
    tau = R_0 * gamma_I / np.mean(degrees)

    from seird import fast_Gillespie
    import matplotlib.pyplot as plt
    fig = plt.figure()
    model = fast_Gillespie(A, tau =tau, gamma_E = gamma_E, gamma_I=gamma_I, tauf=tauf, pd = pd, i0 =i0)
    model.run_sim() # Run the simulation.

    #Plot the aggregate data to see that they make sense
    plt.plot(model.time_grid,N-model.E-model.R-model.D-model.I, color='g', label='S')
    plt.plot(model.time_grid,model.E, color='y', label='E')

    plt.plot(model.time_grid,model.I, color='r', label='I')
    plt.plot(model.time_grid,model.R, color='b', label='R')
    plt.plot(model.time_grid,model.D, color='k', label='D')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('N')
    plt.title('Example of a SBM informed with real-data')

    # plt.savefig("Boston_Example_N_1e4.eps",format='eps')
    plt.savefig(f'./plot_italy/{graph_no}.png')
    plt.close()
    with open(f'./graphs_italy/Adjacency_matrix_edgelist_{graph_no}.csv','w') as the_file:
        for line in nx.generate_edgelist(A, data=False):
            towrite = line.split()
            the_file.write("%d;%d\n"%(int(towrite[0]),int(towrite[1])))

    with open(f'./graphs_italy/Community_labels_{graph_no}.csv','w') as the_file:
        community = 0
        for n in sizes:
            for i in range(n):
                the_file.write("%d\n"%community)
            community+=1
    header = [str(time) for time in model.time_grid]
    #header[0] = "#"+header[0]
    dataframe = pandas.DataFrame(model.nodes_pictures.T)
    dataframe.to_csv(f'./graphs_italy/nodes_frames_{graph_no}.csv',index=False,header=header,sep=';',decimal=',')

'''
# source https://www.tuttitalia.it/statistiche/popolazione-eta-sesso-stato-civile-2020/

census_vector = np.array([
    0.038,      #0-4
    0.044,      #5-9
    0.048,
    0.048,
    0.05,
    0.053,
    0.055,
    0.06,
    0.07,
    0.08,
    0.082,
    0.076,
    0.065,
    0.058,
    0.056,
    0.044,
    0.037+0.023+0.01+0.003])  #85+)   
    
#source https://data.census.gov/cedsci/table?g=0400000US25&tid=ACSST5Y2018.S0101&moe=false&hidePreview=true
census_vector = np.array([
5.3,
5.4,
5.8,
6.7,
7.2,
7.3,
6.8,
6.1,
6.1,
6.7,
7.3,
7.1,
6.3,
5.1,
3.8,
2.6,
2.0,
2.3,])/100  #85+) 
'''