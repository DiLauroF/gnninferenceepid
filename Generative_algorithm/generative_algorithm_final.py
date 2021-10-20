#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 20:30:18 2021

@author: fra
"""
import random
import numpy as np
import networkx as nx
#folder = "contact_matrices/"
#country = "Italy"

#setting = "_country_level_M_overall_contact_matrix_85.csv"
#setting = "_country_level_F_school_setting_85.csv"
pattern = 'United_States_subnational_Massachusetts_M_overall_contact_matrix_85.csv'
class Generate_city():
    def __init__(self,N,contact, populations, percentages,labels):
        self.N = N
        self.contact = contact
        self.populations = populations
        self.percentages = percentages
        self.labels = labels
        self.network = nx.empty_graph(0)
        
    def build_neighborhoods(self): 
        for index,population in enumerate(self.populations):
            p = np.zeros((len(self.percentages),len(self.percentages)))
                        
            for i in range(17):
                for j in range(i):
                    p[i][j] = self.contact[i,j]/(self.percentages[i]*population)
                    p[j][i] = p[i][j]          
            
            sizes = np.round(population*self.percentages).astype(int)

            nodelist= []
            for age in range(len(sizes)):
                for person in range(sizes[age]):
                    nodelist.append("%s_%d_%d"%(self.labels[index],age,person)) 
            
            neigh=nx.stochastic_block_model(sizes,p,nodelist=nodelist)
            self.network = nx.compose(self.network,neigh)
    def connect_neighborhoods(self,neighborhood):
        #To do, find something better
         for i in range(len(self.populations)):
             for j in range(i):
                mean_number_of_contacts = min(self.populations[j].sum(),self.populations[i].sum())/50
                sizes_start = np.round(self.populations[i]*self.percentages).astype(int)
                sizes_end = np.round(self.populations[j]*self.percentages).astype(int)

                p = mean_number_of_contacts/nx.shortest_path_length(neighborhood,self.labels[i],self.labels[j])
                p = p/sizes_end.sum()
                number_of_links = np.random.binomial(sizes_end.sum(),p)
                for link in range(number_of_links):
                    start_node_age = np.random.randint(2,14)
                    end_node_age = np.random.randint(2,14)
                    person_start = np.random.randint(0,sizes_start[start_node_age])
                    person_end =  np.random.randint(0,sizes_end[end_node_age])
                    
                    start_node = "%s_%d_%d"%(self.labels[i],start_node_age,person_start)
                    end_node =  "%s_%d_%d"%(self.labels[j],end_node_age,person_end)  
                    if end_node not in self.network.neighbors(start_node):
                        self.network.add_edge(start_node,end_node)




if __name__ == "__main__":
    #Generate Cambridge
    random.seed(10)
    np.random.seed(10)
    N=50000
    population =np.array([
        10293+43, #1
        556+4303, #2
        6503+13,  #3
        6856+197, #4
        10771+1850, #5
        11394+2044, #6
        7683+4678, #7
        3500+1882,#8
        10674+1360, #9
        8177+426, #10
        13854+97, #11
        1200+132, #12
        2227+120, #13 #END CAMBRIDGE
        22312,   #allston     17
        17577,   #back bay    12
        9305,    #beacon hill 13
        52685,   #brighton
        18058,   #charlestown 19
        #124489,  #dorchester 
        16903,   #downtown    16
        #44989,   #east boston
        32210,   #fenway      11
        #35585,   #hyde park
        #39240,   #jamaica plain
        #5233 ,   #longwood
        #24268,   #mattapan
        #16700,   #misson hill
        9107 ,   #north end   15
        #28644,   #roslindale
        #51252,   #roxbury
        #35660,   #south boston
        #2862 ,   #south boston waterfront
        31601,   #south end   18 
        5945,    #west end    14
        ]) 
    labels = np.array([
        "East Cambridge",
        "MIT",
        "Wellington-Harrington",
        "The Port",
        "Cambridgeport",
        "Mid-Cambridge",
        "Riverside",
        "Agassiz",
        "Neighborhood Nine",
        "West Cambridge",
        "North Cambridge",
        "Cambridge Highlands",
        "Strawberry Hill",    
        'Allston',   #Boston
        'Back bay',
        'Beacon hill',
        'Charlestown',
        'Downtown',
        'Fenway',
        'North end',
        'South end',
        'West end',
        'Brighton'
        ])
    neighborhoods = nx.Graph()
    neighborhoods.add_nodes_from(labels)
    neighborhoods.add_edge('East Cambridge','MIT')
    neighborhoods.add_edge('East Cambridge','Wellington-Harrington')
    neighborhoods.add_edge('MIT','The Port')
    neighborhoods.add_edge('MIT','Cambridgeport')
    neighborhoods.add_edge('Wellington-Harrington','The Port')
    neighborhoods.add_edge('Wellington-Harrington','Mid-Cambridge')
    neighborhoods.add_edge('The Port','Mid-Cambridge')
    neighborhoods.add_edge('The Port','Cambridgeport')
    neighborhoods.add_edge('Cambridgeport','Riverside')
    neighborhoods.add_edge('Mid-Cambridge','Riverside')
    neighborhoods.add_edge('Mid-Cambridge','Agassiz')
    neighborhoods.add_edge('Mid-Cambridge','Neighborhood Nine')
    neighborhoods.add_edge('Mid-Cambridge','West Cambridge')   
    neighborhoods.add_edge('Riverside','West Cambridge')
    neighborhoods.add_edge('Agassiz','Neighborhood Nine')
    neighborhoods.add_edge('Agassiz','North Cambridge')
    neighborhoods.add_edge('Neighborhood Nine','West Cambridge')
    neighborhoods.add_edge('Neighborhood Nine','North Cambridge')
    neighborhoods.add_edge('Neighborhood Nine','Cambridge Highlands')
    neighborhoods.add_edge('West Cambridge','Cambridge Highlands')
    neighborhoods.add_edge('West Cambridge','Strawberry Hill')
    neighborhoods.add_edge('North Cambridge','Cambridge Highlands')
    neighborhoods.add_edge('Cambridge Highlands','Strawberry Hill')
    neighborhoods.add_edge('Downtown','Strawberry Hill')
    neighborhoods.add_edge('Allston','Fenway')
    neighborhoods.add_edge('Brighton','Allston')
    neighborhoods.add_edge('Fenway','Back bay')
    neighborhoods.add_edge('Back bay','South end')
    neighborhoods.add_edge('Back bay','Beacon hill')
    neighborhoods.add_edge('Beacon hill','West end')
    neighborhoods.add_edge('Beacon hill','Downtown')
    neighborhoods.add_edge('West end','North end')
    neighborhoods.add_edge('West end','Downtown')
    neighborhoods.add_edge('West end','Charlestown')
    neighborhoods.add_edge('North end','Charlestown')
    neighborhoods.add_edge('North end','Downtown')
    neighborhoods.add_edge('Downtown','South end')
    neighborhoods.add_edge('Fenway','MIT')
    neighborhoods.add_edge('Allston','Cambridgeport')
    neighborhoods.add_edge('Allston','Riverside')
    neighborhoods.add_edge('Beacon hill','MIT')
    neighborhoods.add_edge('Beacon hill','East Cambridge')
    neighborhoods.add_edge('Charlestown','East Cambridge')
    
    #neighborhoods.add_edge('19','1')
    
    population = np.round(population*N/np.sum(population)).astype(int)

    contact_matrix=np.loadtxt(pattern,ndmin=2,delimiter=',')
    
    agedist = np.loadtxt("United_States_subnational_Massachusetts_age_distribution_85.csv", delimiter =',')
    
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
    
    for i in range(17):
        for j in range(i):
            p[i][j] = contact_matrix[i,j]
            p[j][i] = p[i][j]
    
    Cambridge = Generate_city(N,contact_matrix,population,percentages,labels)           
    Cambridge.build_neighborhoods()
    
    Cambridge.connect_neighborhoods(neighborhoods)
    C = Cambridge.network
    degrees = [C.degree(n) for n in C.nodes()]
    B = nx.relabel.convert_node_labels_to_integers(C, first_label=0, ordering='default', 
                                           label_attribute='provenience')

    
    gamma_E = 1/4   #rate from E to I
    gamma_I = 1/7   #rate from I to R/D
    R_0 = 2.6
    tauf = 700     #final time
    pd = 0.07       #probability of dying
    i0 = 2          #Initially infected nodes
    tau = R_0 * gamma_I / np.mean(degrees) 
    
    from seird import fast_Gillespie
    import matplotlib.pyplot as plt
    fig = plt.figure()             
    model = fast_Gillespie(B, tau =tau, gamma_E = gamma_E, gamma_I=gamma_I, tauf=tauf, pd = pd, i0 =i0, discretestep=tauf)
    model.run_sim() # Run the simulation.
    
    #Plot the aggregate data to see that they make sense
    #plt.plot(model.time_grid,2*N-model.E-model.R-model.D-model.I, color='g', label='S')            
    plt.plot(model.time_grid,model.E/N, color='y', label='E')
    
    plt.plot(model.time_grid,model.I/N, color='r', label='I')
    #plt.plot(model.time_grid,model.R, color='b', label='R')            
    plt.plot(model.time_grid,model.D/N, color='k', label='D')            
    plt.legend()
    plt.show()


    #Save the adjacency matrix as edgelist
    
    with open('Adjacency_matrix_edgelist.csv','w') as the_file:
        for line in nx.generate_edgelist(B, data=False):    
            towrite = line.split()
            the_file.write("%d;%d\n"%(int(towrite[0]),int(towrite[1])))
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
    
 





















#else:
    '''
    pattern = folder+country+setting
    contact_matrix=np.loadtxt(pattern,ndmin=2,delimiter=',')
    
    agedist = np.loadtxt("age_distributions/Italy_country_level_age_distribution_85.csv", delimiter =',')
    
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
    
    for i in range(85):
        for j in range(i):
            if (agedist[i,1]*contact_matrix[i,j] - agedist[j,1]*contact_matrix[j,i])>1:
                print(i,j)
        
    
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
    N=20000
    
    for i in range(17):
        for j in range(i):
            p[i][j] = contact_matrix[i,j]/(percentages[i]*N)
            p[j][i] = p[i][j]
    
    
    sizes = np.round(N*percentages).astype(int)
    
    import networkx as nx
    #p = contact_matrix/np.reshape(np.outer(np.sqrt(sizes),np.sqrt(sizes)),(17,17))
    
    #Generate Cambridge
    population =np.array([
        10293+43, #1
        556+4303, #2
        6503+13,  #3
        6856+197, #4
        10771+1850, #5
        11394+2044, #6
        7683+4678, #7
        3500+1882,#8
        10674+1360, #9
        8177+426, #10
        13854+97, #11
        1200+132, #12
        2227+120]) #13
    labels = np.array([
        "East Cambridge",
        "MIT",
        "Wellington-Harrington",
        "The Port",
        "Cambridgeport",
        "Mid-Cambridge",
        "Riverside",
        "Agassiz",
        "Neighborhood Nine",
        "West Cambridge",
        "North Cambridge",
        "Cambridge Highlands",
        "Strawberry Hill",    
        ])
    
    population = np.round(population*N/np.sum(population)).astype(int)
    
    
    
    #Generate Boston
    index_min = len(C)
    N=20000
    population =np.array([
         22312,   #allston     17
         17577,   #back bay    12
         9305,    #beacon hill 13
         52685,   #brighton
         18058,   #charlestown
         124489,  #dorchester 
         16903,   #downtown    16
         44989,   #east boston
         32210,   #fenway      11
         35585,   #hyde park
         39240,   #jamaica plain
         5233 ,   #longwood
         24268,   #mattapan
         16700,   #misson hill
         9107 ,   #north end   15
         28644,   #roslindale
         51252,   #roxbury
         35660,   #south boston
         2862 ,   #south boston waterfront
         31601,   #south end   18 
         5945,    #west end    14
         32795,   #west roxbury
        ])
    population = np.round(population/np.sum(population)*N).astype(int)
    
    Boston_matrices=[]
    populations=[]
    for i in range(len(population)):
        sizes = np.round(population[i]*percentages).astype(int)
        
        if sizes.sum()>population[i]:
            sizes[0] = sizes[0] -1
        if sizes.sum()<population[i]:
            sizes[0] +=1 
        populations.append(sizes.sum())
        if i>=1:
            Boston_matrices.append(nx.stochastic_block_model(sizes,p,nodelist=range(len(C),len(C)+sizes.sum())))
    
            C = nx.compose(C,Boston_matrices[-1])
            print(len(C))
        else:       
            B=nx.stochastic_block_model(sizes,p,nodelist=range(len(C),len(C)+sizes.sum()))
            C = nx.compose(C,B)
    
    communities = np.cumsum(populations)
    for i in range(len(populations)):
        for j in range(i):
            number_of_links = np.random.randint(5,100)
            for link in range(number_of_links):
                start_node = np.random.randint(populations[i])
                end_node =  np.random.randint(populations[j])
                if i>0:
                    start_node += communities[i-1]
                if j>0:
                    end_node += communities[j-1]
                C.add_edge(start_node,end_node)
    
    
    #Connect cities
    
    
    
    
    
    
    
    
    xls = pd.ExcelFile("zones.xlsx")
    df = pd.read_excel(xls,header=None,NaN=0)
    df=df.fillna(0)
    contact_matrix = df.to_numpy()
    contact_matrix = contact_matrix + contact_matrix.T
    
    
    
    
    
    
    degrees = [C.degree(n) for n in C.nodes()]
    B = nx.relabel.convert_node_labels_to_integers(C, first_label=0, ordering='default', 
                                           label_attribute='provenience')
    
    gamma_E = 1/4   #rate from E to I
    gamma_I = 1/7   #rate from I to R/D
    R_0 = 3
    tauf = 700     #final time
    pd = 0.07       #probability of dying
    i0 = 4          #Initially infected nodes
    tau = R_0 * gamma_I / np.mean(degrees) 
    
    from seird import fast_Gillespie
    import matplotlib.pyplot as plt
    fig = plt.figure()             
    model = fast_Gillespie(B, tau =tau, gamma_E = gamma_E, gamma_I=gamma_I, tauf=tauf, pd = pd, i0 =i0)
    model.run_sim() # Run the simulation.
    
    #Plot the aggregate data to see that they make sense
    #plt.plot(model.time_grid,2*N-model.E-model.R-model.D-model.I, color='g', label='S')            
    plt.plot(model.time_grid,model.E, color='y', label='E')
    
    plt.plot(model.time_grid,model.I, color='r', label='I')
    #plt.plot(model.time_grid,model.R, color='b', label='R')            
    plt.plot(model.time_grid,model.D, color='k', label='D')            
    plt.legend()
    plt.show()
    
    
    
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
