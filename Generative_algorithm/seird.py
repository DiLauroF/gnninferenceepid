#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 13:43:56 2020

@author: fra
"""
import networkx as nx
import numpy as np
from heapq import *
from datetime import datetime

#0:00:23.466188
class Node():
    def __init__(self,index,status, time):
        self.index = index
        self.status = status
        self.rec_time = time
class Event():
    def __init__(self,node,time,action, source=None):
        self.time = time

        self.node = node
        self.action = action
        self.source=source
    def __lt__(self, other):
        '''
            This is read by heappush to understand what the heap should be about
        '''
        return self.time < other.time        

class fast_Gillespie():
    def __init__(self, A, tau=1.0, gamma_E=1.0, gamma_I=1.0, i0=10, tauf=10, pd=0, discretestep=500):
        if type(A)==nx.classes.graph.Graph:
            self.N = nx.number_of_nodes(A)
            self.A = A
        else:
            raise BaseException("Input networkx object only.")
        labels= np.array(['susceptible','exposed','infected','recovered','dead'])
        namelabels = np.arange(0,len(labels),1)
        self.dictionary = dict(zip(labels, namelabels))
        # Model Parameters (See Istvan paper).
        self.tau = tau
        self.gamma_E = gamma_E
        self.gamma_I = gamma_I
        self.pd = pd
        self.tauf = tauf
        
        # Time-keeping.
        self.cur_time = 0
        #output time vector
        self.time_grid =np.linspace(0,tauf,discretestep)
        self.current_index=0
        self.discretestep = discretestep
        #Node numbers.
        self.I = np.zeros(discretestep)
        self.E = np.zeros(discretestep)
        self.D = np.zeros(discretestep)
        self.R = np.zeros(discretestep)
        self.nodes_pictures = np.zeros((discretestep,self.N),dtype='str')
        
        #number of SI links
        self.SI=np.zeros(self.N+1)
        #time in each state
        self.tk = np.zeros(self.N+1)
        
        #node state is [0] if not infected and [1] if infected
        X = np.array([0]*(self.N-i0) +[1]*i0)
        #nodes initialisation
        self.nodes = [Node(i,'susceptible', 0) for i in range(self.N)] 
        
        #keeps count of how many infected, useful for self.I and self.SI updates
        self.num_I = 0
        self.num_E = 0
        self.num_R = 0
        self.num_D = 0

        #display randomly the initial infected nodes
        np.random.shuffle(X)
        #Queue of Events, here each node has its own event
        self.queue=[]
        self.times=[]
        self.infected=[]
        self.cur_time=0
        for index in np.where(X==1)[0]:
            event = Event(self.nodes[index],0,'become_I', source=Node(-1,'infected',0))
            heappush(self.queue,event)
        
    def run_sim(self):
        '''first round outside to determine SI'''
        num_SI=0        
        while self.queue:
            '''
                condition to stop
            '''
            event = heappop(self.queue)
            #dt is used only to update SI
            '''
            If node is susceptible and it has an event it must be an infection
            '''
            if event.action=='become_I':
                if event.node.status =='exposed' or event.node.status=='susceptible':
                    dt = event.time -self.cur_time
                    #set new time accordingly
        
                    '''
                    check if time grid needs to be updated
                    '''
                    if self.cur_time <self.tauf:
                        while self.time_grid[self.current_index] <= self.cur_time:                    
                            self.I[self.current_index] = self.num_I
                            self.D[self.current_index] = self.num_D
                            self.E[self.current_index] = self.num_E
                            self.R[self.current_index] = self.num_R
                            self.nodes_pictures[self.current_index] = np.array([self.dictionary[node.status] for node in self.nodes])
                            self.current_index +=1      
                    
                    '''
                    AFTER finding dt you can update SI
                    '''
                    self.SI[self.num_I] += num_SI*dt
                    self.tk[self.num_I] += dt
                    num_SI +=self.process_trans(event.node, event.time)                           
            
            elif event.action=='become_E':
                if event.node.status =='susceptible':
                    if self.cur_time <self.tauf:
                        while self.time_grid[self.current_index] <= self.cur_time:                    
                            self.I[self.current_index] = self.num_I
                            self.D[self.current_index] = self.num_D
                            self.E[self.current_index] = self.num_E
                            self.R[self.current_index] = self.num_R 
                            self.nodes_pictures[self.current_index] = np.array([self.dictionary[node.status] for node in self.nodes])
                            self.current_index +=1      
                        dt = event.time -self.cur_time
                        self.SI[self.num_I] += num_SI*dt
                        self.tk[self.num_I] += dt
                        num_SI +=self.process_infect(event.node, event.time)                
                    
            else:
                if event.node.status =='infected':
               
                    if self.cur_time <self.tauf:
                        while self.time_grid[self.current_index] <= self.cur_time:                    
                            self.I[self.current_index] = self.num_I
                            self.D[self.current_index] = self.num_D
                            self.E[self.current_index] = self.num_E
                            self.R[self.current_index] = self.num_R                        
                            self.nodes_pictures[self.current_index] = np.array([self.dictionary[node.status] for node in self.nodes])
                            self.current_index +=1      
                        dt = event.time -self.cur_time
                        self.SI[self.num_I] += num_SI*dt
                        self.tk[self.num_I] += dt
                    num_SI +=self.process_rec(event.node,event.time)
        
        if self.current_index < len(self.I):
            self.I[self.current_index:] = self.num_I
            self.D[self.current_index:] = self.num_D
            self.E[self.current_index:] = self.num_E
            self.R[self.current_index:] = self.num_R                        
            status = np.array([self.dictionary[node.status] for node in self.nodes])
            self.nodes_pictures[self.current_index:] = status
           
                
    def process_trans(self,node,time):
        '''
        utility for transmission events:
        it checks also the neighbours.
        Returns number of SI as well
        '''
        #self.times.append(time)
        self.cur_time=time
        self.num_I +=1
        if node.status != 'susceptible':
            self.num_E -=1
        '''
        if len(self.infected) >0:
            self.infected.append(self.infected[-1]+1)
        else:
            self.infected.append(1)
        '''    
        node.status='infected'
        
        r1 = np.random.rand()
        rec_time = time -1.0/self.gamma_I *np.log(r1)
        node.rec_time = rec_time
        
        if rec_time < self.tauf:
            event = Event(node,rec_time,'recover', None)
            heappush(self.queue,event)
        num_SI=0    
        for index in self.A.neighbors(node.index):
            neighbor = self.nodes[index]
            if neighbor.status=='susceptible':
                num_SI+=1
            else:
                num_SI-=1
            self.find_next_trans(source = node, target = neighbor, time = time)
        return num_SI
    def find_next_trans(self,source,target,time):
        if target.rec_time < source.rec_time:
            r1 = np.random.rand()
            trans_time = max(time,target.rec_time) -1.0/self.tau *np.log(r1)
            if trans_time < source.rec_time and trans_time<self.tauf:
                event = Event(node=target, time=trans_time,  action='become_E', source=source)
                heappush(self.queue,event)
                
    def process_rec(self, node, time):
        chance = np.random.uniform()
        if chance <=self.pd:
            node.status='dead'
            self.num_D +=1
        else:
            node.status='recovered'
            self.num_R +=1
        node.rec_time = 0
        num_SI=0
        self.num_I -=1
        for index in self.A.neighbors(node.index):
            neighbor = self.nodes[index]
            if neighbor.status=='susceptible':
                num_SI-=1
        #self.times.append(time)
        self.cur_time=time
        #self.infected.append(self.infected[-1]-1)
        return num_SI
    def process_infect(self, node, time):
        node.status='exposed'
        node.rec_time = 0
        num_SI=0
        self.num_E +=1
        for index in self.A.neighbors(node.index):
            neighbor = self.nodes[index]
            if neighbor.status=='infected':
                num_SI-=1
        #self.times.append(time)
        self.cur_time=time
        r1 = np.random.rand()
        rec_time = time -1.0/self.gamma_E *np.log(r1)
        node.rec_time = rec_time
        
        if rec_time < self.tauf:
            event = Event(node,rec_time,'become_I', None)
            heappush(self.queue,event)
        #self.infected.append(self.infected[-1]-1)
        return num_SI                        

if __name__=="__main__":
    #np.random.seed(12)
    #import random
    #random.seed(11)



    from matplotlib import pyplot as plt
    
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