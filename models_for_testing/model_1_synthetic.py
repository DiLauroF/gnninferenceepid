#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 09:51:10 2021

@author: fra
"""

import seird
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__=="__main__":
    #np.random.seed(12)
    #import random
    #random.seed(11)
    '''
    This model generates a completely synthetic network with 4 communities of
    different sizes. IT IS NOT BASED ON REAL DATA and it is intended as a simple
    benchmark.
    '''
    
    N = 10000 #network size
    k = 19   #network average degree (We are using Erdos Renyi so far)
    gamma_E = 1/4   #rate from E to I
    gamma_I = 1/7   #rate from I to R/D
    tauf = 250      #final time
    pd = 0.07       #probability of dying
    i0 = 4          #Initially infected nodes
    #print(tauf)
    networkchoice='E-R'   
    
    #Stochastic block-model
    n_blocks = 5    #number of communities
    
    sizes = np.random.multinomial(N,np.random.dirichlet(np.ones(5)*4,size=1)[0])
    #The size of each community will be random, with a good degree of heterogeneity
    #note to self: in the future, take age distribution instead of dirichlet
    
    p = np.diag((k-4)/sizes)/2  #On average, each node will have 10 neighbours in the same community
    p[0][1] = 1/(sizes[1]*sizes[0])
    p[0][2] = 2/(sizes[2]*sizes[0])
    p[0][3] = 3/(sizes[3]*sizes[0])
    p[0][4] = 2/(sizes[0]*sizes[4])
    p[1][2] = 4/(sizes[1]*sizes[2])
    p[1][3] = 2/(sizes[1]*sizes[3])
    p[1][4] = 2/(sizes[1]*sizes[4])    
    p[2][3] = 5/(sizes[2]*sizes[3])
    p[2][4] = 2/(sizes[2]*sizes[4])
    p[3][4] = 1/(sizes[3]*sizes[4])
    
    p = p + p.T
    A = nx.stochastic_block_model(sizes,p.tolist())
    degrees = [A.degree(n) for n in A.nodes()]
    #print(np.mean(degrees))
    tau = 0.4/np.mean(degrees)     #infection parameter
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
    plt.show()
    
    #Save the adjacency matrix as edgelist
    
    with open('Adjacency_matrix_edgelist.csv','w') as the_file:
        for line in nx.generate_edgelist(A, data=False):    
            towrite = line.split()
            the_file.write("%d;%d\n"%(int(towrite[0]),int(towrite[1])))
    with open('Community_labels.csv','w') as the_file:
        community = 0
        for n in sizes:
            for i in range(n):
                the_file.write("%d\n"%community)
            community+=1
    #save nodes statuses:
    #n.b. each column will be a different time and each line is the status of the network
    
    #n.b. this is how to interpret the numbers:
    # 0 - susceptible
    # 1 - exposed
    # 2 - infected
    # 3 - recovered
    # 4 - dead
    
    header = [str(time) for time in model.time_grid]
    #header[0] = "#"+header[0]
    import pandas as pd
    dataframe = pd.DataFrame(model.nodes_pictures.T)
    dataframe.to_csv('nodes_frames.csv',index=False,header=header,sep=';',decimal=',')