# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 15:55:54 2020

@author: Matteo
"""

# from sklearn.preprocessing import LabelEncoder
import torch
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np
import random
import time
import math
random.seed(1)
import os
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv
from torch_geometric import utils
from torch_geometric.data import GraphSAINTRandomWalkSampler
from matplotlib import pyplot as plt
import sys

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix


from gnn_seird_dataloader import load_graphs, load_graphs_italy, balance_data, balance_data_italy,balance_data_scenario1
from torch_geometric.data import InMemoryDataset
torch.cuda.empty_cache()

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

######## CLASS

class BasicBlock(torch.nn.Module):
    def __init__(self, in_f, out_f, dropout):
        super(BasicBlock, self).__init__()
        self.conv = GCNConv(in_f, out_f)
        self.relu = nn.ReLU()
        self.dropout  = nn.Dropout(dropout)

    def forward(self, _input):
        _input.x = self.conv(_input.x,_input.edge_index)
        _input.x = self.relu(_input.x)
        _input.x= self.dropout(_input.x)
        return _input

def GNN_block(in_f, out_f, dropout, **kwargs):
    return nn.Sequential(
        GCNConv(in_f, out_f),
        # nn.ReLU(inplace=True),
        # #distinction b/w training and eval
        # nn.Dropout(dropout,inplace=True)
    )


class simple_net(torch.nn.Module):
    def __init__(self, hidden_channels,dropout,no_layers=2,num_features=5):
        # Init parent
        super(simple_net, self).__init__()
        torch.manual_seed(42)
        self.dropout = dropout

        # GCN layers: 2 message passing layers (to create embedding)
        self.conv1 = Linear(num_features, hidden_channels)
        self.conv2 = Linear(hidden_channels, hidden_channels)

        # Output layer: is the classification layer
        self.out = Linear(hidden_channels, num_features)

    def forward(self, _input):
        x, edge_index, batch = _input.x, _input.edge_index, _input.batch
        # First Message Passing layer: is equal as in NN with the edge info
        x = self.conv1(x)
        x = x.relu() # the classical activation function
        x = F.dropout(x, p=self.dropout, training=self.training)# and dropout to avoid overfitting

        # Second Message Passing layer
        x = self.conv2(x)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layer: as in NN output activation function with a probability (to be a certain class) as ouput
        x = F.softmax(self.out(x), dim=1) #is the classification layer
        return x

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels,dropout,no_layers = 2,num_features=5):
        # Init parent
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.dropout = dropout

        # GCN layers: 2 message passing layers (to create embedding)
        self.conv1 = GCNConv(num_features, hidden_channels, add_self_loops=True)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        # Output layer: is the classification layer
        self.out = Linear(hidden_channels, num_features)

    def forward(self, _input):
        x, edge_index, batch = _input.x, _input.edge_index, _input.batch
        # First Message Passing layer: is equal as in NN with the edge info
        x = self.conv1(x, edge_index)
        x = x.relu() # the classical activation function
        x = F.dropout(x, p=self.dropout, training=self.training)# and dropout to avoid overfitting

        # Second Message Passing layer
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layer: as in NN output activation function with a probability (to be a certain class) as ouput
        x = F.softmax(self.out(x), dim=1) #is the classification layer
        return x

class GCN1(torch.nn.Module):
    def __init__(self, hidden_channels,dropout,no_layers = 2,num_features=5):
        # Init parent
        super(GCN1, self).__init__()
        torch.manual_seed(42)
        self.dropout = dropout

        # GCN layers: 2 message passing layers (to create embedding)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)

        # Output layer: is the classification layer
        self.out = Linear(hidden_channels, num_features)

    def forward(self, _input):
        x, edge_index, batch = _input.x, _input.edge_index, _input.batch
        # First Message Passing layer: is equal as in NN with the edge info
        x = self.conv1(x, edge_index)
        x = x.relu() # the classical activation function
        x = F.dropout(x, p=self.dropout, training=self.training)# and dropout to avoid overfitting

        # Second Message Passing layer
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Third Message Passing layer
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 4th Message Passing layer
        x = self.conv4(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layer: as in NN output activation function with a probability (to be a certain class) as ouput
        x = F.softmax(self.out(x), dim=1) #is the classification layer
        return x

class GCN_l(torch.nn.Module):
    def __init__(self, hidden_channels, dropout, no_layers = 2, num_features=5):
        # Init parent
        super(GCN_l, self).__init__()
        torch.manual_seed(42)
        self.dropout = dropout
        self.no_layers = no_layers
        # GCN layers: 2 message passing layers (to create embedding)
        module = nn.ModuleList([BasicBlock(num_features, hidden_channels,self.dropout)])
        for layer in range(no_layers):
            module.append(BasicBlock(hidden_channels, hidden_channels,self.dropout))

        self.conv = nn.Sequential(*module)

        # Output layer: is the classification layer
        self.out = Linear(hidden_channels, num_features)

    def forward(self, _input):
        x, edge_index, batch = _input.x, _input.edge_index, _input.batch
        _input = self.conv(_input)

        # Output layer: as in NN output activation function with a probability (to be a certain class) as ouput
        x = F.softmax(self.out(_input.x), dim=1) #is the classification layer
        return x

class Recurrent_layer_GCN(torch.nn.Module):
    def __init__(self, hidden_channels,dropout):
        # Init parent
        super(Recurrent_layer_GCN, self).__init__()
        torch.manual_seed(42)
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        self.lookback = 3

        self.projection = nn.Linear(num_features, hidden_channels)

        # GCN layers: 2 message passing layers (to create embedding)
        self.conv1 = GCNConv(self.hidden_channels, self.hidden_channels)
        self.conv2 = GCNConv(self.hidden_channels, self.hidden_channels)

        self.layer_count = 1
        self.hidden_c = nn.Parameter(torch.randn(self.layer_count, 1, self.hidden_channels).cuda(), requires_grad=True)
        self.Recurrent_unit1 = nn.GRUCell(self.hidden_channels, self.hidden_channels)

        # Output layer: is the classification layer
        self.out = Linear(hidden_channels, 5)

    def forward(self, _input):
        x, edge_index, batch = _input.x, _input.edge_index, _input.batch
        # First Message Passing layer: is equal as in NN with the edge info

        for k_lookback in range(self.lookback):
            if k_lookback == 0:
                next_hidden = self.init_hidden(batch_size=x.shape[0])[0]
                x_seq = self.projection(x)
                next_hidden = self.Recurrent_unit1(x_seq, next_hidden)

            #I am not using current graph node features-rectify
            x_seq = self.conv1(next_hidden, edge_index)
            x_seq = x_seq.relu() # the classical activation function
            x_seq = F.dropout(x_seq, p=self.dropout, training=self.training)# and dropout to avoid overfitting

            # Second Message Passing layer
            x_seq = self.conv2(x_seq, edge_index)
            x_seq = x_seq.relu()
            x_seq = F.dropout(x_seq, p=self.dropout, training=self.training)

            #RNN
            next_hidden = self.Recurrent_unit1(x_seq, next_hidden)

        # Output layer: as in NN output activation function with a probability (to be a certain class) as ouput
        x_seq = F.softmax(self.out(next_hidden), dim=1) #is the classification layer

        return x_seq

    def init_hidden(self, batch_size=1):
        return self.hidden_c.expand(self.layer_count, batch_size, self.hidden_channels).contiguous()

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels,dropout,no_layers = 2,num_features=5):
        # Init parent
        super(GAT, self).__init__()
        torch.manual_seed(42)
        self.dropout = dropout

        # GCN layers: 2 message passing layers (to create embedding)
        self.conv1 = GATConv(num_features, hidden_channels, add_self_loops=True)
        self.conv2 = GATConv(hidden_channels, hidden_channels)

        # Output layer: is the classification layer
        self.out = Linear(hidden_channels, num_features)

    def forward(self, _input):
        x, edge_index, batch = _input.x, _input.edge_index, _input.batch
        # First Message Passing layer: is equal as in NN with the edge info
        x = self.conv1(x, edge_index)
        x = x.relu() # the classical activation function
        x = F.dropout(x, p=self.dropout, training=self.training)# and dropout to avoid overfitting

        # Second Message Passing layer
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layer: as in NN output activation function with a probability (to be a certain class) as ouput
        x = F.softmax(self.out(x), dim=1) #is the classification layer
        return x

######## FUNCTIONS


def train(_train_data):
    loss_all = 0
    total_graphs = 0
    for k, _data in enumerate(_train_data):
        #Uncomment it if using Inmemory data
        _data = _data.to(device)
        model.train() # is the function of the parent class
        optimizer.zero_grad() # Reset gradients
        # use all data as input, because all nodes have node features
        out = model(_data)
        loss = criterion(out, _data.y)
        loss.backward()
        optimizer.step()
        loss_all += _data.num_graphs * loss.item()
        total_graphs += _data.num_graphs
    return loss_all / total_graphs


def test(_test_data):
    test_correct = 0
    test_nodes = 0
    y_true = np.array([])
    y_pred = np.array([])
    with torch.no_grad():
        for k, _data in enumerate(_test_data):
            _data = _data.to(device)
            model.eval() # is the parent function
            out = model(_data)
            # use the classes with highest probability
            pred = out.argmax(dim=1)
            node_state = np.hstack(_data.state)
            unknown_indexes = np.where(node_state == 0)[0]
            pred_correct = (pred[unknown_indexes] == _data.y[unknown_indexes])
            test_correct += pred_correct.sum()
            test_nodes += unknown_indexes.shape[0]
            y_true = np.hstack((y_true, _data.y[unknown_indexes].cpu().detach().numpy()))
            y_pred = np.hstack((y_pred, pred[unknown_indexes].cpu().detach().numpy()))

    cm = confusion_matrix(y_true, y_pred)  # (y_true,y_prediction)
    test_acc = int(test_correct) / test_nodes
    return test_acc,cm

def load_data(path, data_frame, total_graphs, num_days, num_nodes,device):
    ###change the data loader
    x = torch.tensor(data_frame.values[:,2:5], dtype=torch.float)
    y = torch.tensor(data_frame['y'], dtype=torch.long)
    data_list = []
    for j in range(total_graphs):
        loc = os.path.join(path,f'Adjacency_matrix_edgelist_{j}.csv')
        edges = pd.read_csv(loc, header=None, sep=';').to_numpy()
        # Edges list: in the format (head, tail); the order is irrelevant
        edge_index = utils.to_undirected(torch.tensor(edges, dtype=torch.long).t())
        for i in range(num_days):
            start_idx = i*num_nodes + j*num_nodes*num_days
            end_idx = i*num_nodes+ j*num_nodes*num_days + num_nodes
            if data_frame.use_data[start_idx] ==1:
                data_list.append(Data(x=x[start_idx:end_idx,:], y=y[start_idx:end_idx],
                                      edge_index=edge_index, state=data_frame.state[start_idx:end_idx]))

    return data_list

######## PREPROCESS DATA

class GNNDataset(InMemoryDataset):
    def __init__(self, root,path_data, transform=None, pre_transform=None):
        super(GNNDataset, self).__init__(root,transform, pre_transform)
        self.df = df_test
        self.path = path_data
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['../data/chk.dataset']

    def download(self):
        pass

    def process(self):
        ####cHANGE THIS TO ACCEPT BOTH TEST AND TRAIN PATHS:df and path
        path = './graphs_rough'
        data_list = []
        x = torch.tensor(df_test.values[:,2:7], dtype=torch.float)
        y = torch.tensor(df_test['y'], dtype=torch.long)
        for j in range(no_graphs_test):
            loc = os.path.join(path,f'Adjacency_matrix_edgelist_{j}.csv')
            edges = pd.read_csv(loc, header=None, sep=';').to_numpy()
            # Edges list: in the format (head, tail); the order is irrelevant
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            for i in range(num_days):
                start_idx = i*num_nodes + j*num_nodes*num_days
                end_idx = i*num_nodes+ j*num_nodes*num_days + num_nodes
                data_list.append(Data(x=x[start_idx:end_idx,:], y=y[start_idx:end_idx],
                                      edge_index=edge_index, state=df_test.state[start_idx:end_idx]))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# Load data
if __name__=="__main__":
    weight_r = 1.0
    weight_i = 1.0
    weight_s = 1.0
    # hidden_nodes = 0.8
    layers = 2
    k_days = 15
    hidden_units = 64
    dropout_hidden = 0.4
    layer_loop = 'GCN'
    hyper_parameters = zip([0.8])
    #For undersmapling define no. of graphs from staring days
    #model1-cyoff:30, data_pt-3, italy:cutoffday-25, boston:100
    cuttoff_day = 20
    no_data_point = 3
    no_data_point_test = 5
    num_days = 90

    model_save_dir = './save_mode1_reduced_beta'
    if not os.path.isdir(model_save_dir):
        os.mkdir(model_save_dir)
    loop = 0
    for hidden_nodes in [0.8,0.9,0.95]:
        loop +=1
        # file_to_write = open(os.path.join(model_save_dir,f"hyperparameter_tuning_{loop}.txt" ),"w")
        print(f'Testing:Scenario 1 with reduced beta and undersampling both sides')
        no_graphs = 80
        no_graphs_test = 40

        print(f'Train graphs:{no_graphs},Test graphs:{no_graphs_test},'
              f'hidden_nodes:{hidden_nodes}, k_days:{k_days}, dropout = {dropout_hidden}')
        print(f'weight of exposed class:{weight_i}\n'
              f'weight of recovered class:{weight_r}\n'
              f'weight of infected class:{weight_i}')
        file_loc_train = './graphs_1_beta'
        file_loc_test = './graphs_1_beta_test'
        print(f'location of graphs:{file_loc_train}, dropout= {dropout_hidden}')
        start_idx_graph = 0
        df, num_nodes, num_days = load_graphs_italy(no_graphs, hidden_nodes, file_loc_train, start_idx_graph,k_days,num_days)
        print(f'No.of_days_used:{num_days}, No.of nodes={num_nodes}')

        df = balance_data_scenario1(df, num_days, num_nodes, no_graphs, cuttoff_day)
        # feature scaling
        sc = MinMaxScaler()
        known_idx_train = np.where(df['state'] == 1)[0]
        scaler = sc.fit(df.values[known_idx_train,2:5])
        df.values[known_idx_train,2:5] = scaler.transform(df.values[known_idx_train,2:5])


        # use GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        train_data_list = load_data(file_loc_train, df, no_graphs, num_days, num_nodes, device)
        num_features = train_data_list[-1].num_features
        train_indices = [id for id in range(len(train_data_list))]
        random.shuffle(train_indices)
        train_dataset = DataLoader(train_data_list, batch_size=256, sampler=train_indices)

        num_features = 3
        # train_dataset.num_classes = 5 # S,E,I,R,D

        # Initialize Model
        if layer_loop == 'GCN':
            model = GCN_l(hidden_channels=hidden_units, dropout=dropout_hidden, no_layers=layers,num_features= num_features)
        else:
            model = GAT(hidden_channels=32, dropout=dropout_hidden, no_layers=layers,num_features= num_features)
        print(model)
        model = model.to(device)

        #0.0005 for italy
        learning_rate = 0.0002 # step for gradient descendent method for learning (?)
        decay = 5e-4 #decay of importance of learning rate
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)

        # Define loss function (CrossEntropyLoss for Classification problems with probability distribution)
        #for model 1 use_dat was not used
        use_state_idx = np.where(df['use_data'] == 1)[0]
        w0 = np.where(df['y'][use_state_idx] == 0)[0].shape[0]
        w1 = np.where(df['y'][use_state_idx] == 1)[0].shape[0]
        w2 = np.where(df['y'][use_state_idx] == 2)[0].shape[0]
        w3 = np.where(df['y'][use_state_idx] == 3)[0].shape[0]
        w4 = np.where(df['y'][use_state_idx] == 4)[0].shape[0]
        max_w = max(w0,w1,w2,w3,w4)
        # weight = torch.tensor([(max_w/w0)*weight_s,(max_w/w1)*weight_scale_1,(max_w/w2)*weight_scale,(max_w/w3)*weight_r,(max_w/w4)]).to(device)
        #On combing exposed and infected class
        weight = torch.tensor([(max_w/w0)*weight_s,(max_w/w1)*weight_i,(max_w/w2)*weight_r]).to(device)

        criterion = torch.nn.CrossEntropyLoss(weight=weight)
        print(weight)
        # cross entropy compare probabilities, and we have probabilities because of softmax

        # Train !
        del df
        del train_data_list
        losses = []
        start = time.time()
        for epoch in range(1,251):
            loss = train(train_dataset)
            losses.append(loss)
            if epoch % 50== 0:
                test_acc, conf_matrix = test(train_dataset)
                infected_acc = conf_matrix[2,2]/np.sum(conf_matrix[2,:])
                infected_precision = conf_matrix[2,2]/np.sum(conf_matrix[:,2])
                F1_infected = 2* (infected_acc*infected_precision)/(infected_acc+infected_precision)
                print(f'[{time_since(start)}], Epoch: {epoch:03d}, Loss: {loss:.4f},tot_acc={test_acc}, F1_inf:{F1_infected}')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'scaler': scaler,
        }, os.path.join(model_save_dir,f'model_scenario1_{loop}'))

        # Train results
        import seaborn as sns
        losses_float = [loss for loss in losses]
        loss_indices = [i for i,l in enumerate(losses_float)]
        sns.lineplot(loss_indices, losses_float)
        plt.savefig(os.path.join(model_save_dir,f'loss_scenario1_{loop}'))

        # ######################################################
        # Test the model with different unknown nodes

        del train_dataset
        total_acc_all = np.array([])
        acc_infected_all = np.array([])
        acc_susceptible_all = np.array([])
        acc_recovered_all = np.array([])
        precision_susceptible_all = np.array([])
        precision_infected_all = np.array([])
        precision_recovered_all = np.array([])
        F1_infected_all = np.array([])
        balanced_accuracy_all = np.array([])
        for n_test in range(1):
            torch.cuda.empty_cache()
            start_idx_graph = n_test * no_graphs_test
            df_test, num_nodes, num_days = load_graphs_italy(no_graphs_test, hidden_nodes,
                                                             file_loc_test,start_idx_graph, k_days,num_days)

            # df_test = balance_data_italy(df_test, num_days, num_nodes, no_graphs_test, cuttoff_day)
            known_idx_test = np.where(df_test['state'] == 1)[0]
            df_test.values[known_idx_test,2:5] = scaler.transform(df_test.values[known_idx_test,2:5])
            test_data_list = load_data(file_loc_test,df_test, no_graphs_test, num_days, num_nodes,device)
            test_indices = [id for id in range(len(test_data_list))]
            test_dataset = DataLoader(test_data_list, batch_size=1, sampler=test_indices)

            test_acc, conf_matrix = test(test_dataset)
            print(f'Test_set:{n_test}, Test Accuracy: {test_acc:.4f},\n'
                  f' confusion_matrix:{conf_matrix}')

            #change here if infected and exposed are combined
            susceptible_acc = conf_matrix[0,0]/np.sum(conf_matrix[0,:])
            infected_acc = conf_matrix[1,1]/np.sum(conf_matrix[1,:])
            recovered_acc = conf_matrix[2,2]/np.sum(conf_matrix[2,:])
            susceptible_precision = conf_matrix[0,0]/np.sum(conf_matrix[:,0])
            infected_precision = conf_matrix[1,1]/np.sum(conf_matrix[:,1])
            recovered_precision = conf_matrix[2,2]/np.sum(conf_matrix[:,2])

            F1_infected = 2*(infected_acc*infected_precision)/(infected_acc+infected_precision)

            balanced_accuracy = 1/3 * (conf_matrix[0,0]/np.sum(conf_matrix[0,:])+
                                       conf_matrix[1,1]/np.sum(conf_matrix[1,:])+
                                       conf_matrix[2,2]/np.sum(conf_matrix[2,:]))

            print(f'Accuracy_infected = {infected_acc}\n'
                  f'Precision_infected = {infected_precision}\n'
                  f'F1_score_infected = {F1_infected}\n'
                  f'Balanced accuracy = {balanced_accuracy}')
            print(f'Conf_matrix_percentage:\n'
                  f'{conf_matrix/conf_matrix.sum(axis=1)[:,None]}')

            total_acc_all = np.append(total_acc_all, test_acc)
            acc_susceptible_all = np.append(acc_susceptible_all,susceptible_acc)
            acc_infected_all = np.append(acc_infected_all,infected_acc)
            acc_recovered_all = np.append(acc_recovered_all,recovered_acc)
            precision_susceptible_all = np.append(precision_susceptible_all,susceptible_precision)
            precision_infected_all = np.append(precision_infected_all,infected_precision)
            precision_recovered_all = np.append(precision_recovered_all,recovered_precision)
            F1_infected_all = np.append(F1_infected_all,F1_infected)
            balanced_accuracy_all = np.append(balanced_accuracy_all,balanced_accuracy)

            del df_test
            del test_data_list
            del test_indices
            del test_dataset

        print(f'Mean values:\n'
              f'Total accuracy = {total_acc_all.mean()}\n'
              f'Accuracy_susceptible = {acc_susceptible_all.mean()}\n'
              f'Accuracy_infected = {acc_infected_all.mean()}\n'
              f'Accuracy_recovered = {acc_recovered_all.mean()}\n'
              f'Precision_susceptible = {precision_susceptible_all.mean()}\n'
              f'Precision_infected = {precision_infected_all.mean()}\n'
              f'Precision_recovered = {precision_recovered_all.mean()}\n'
              f'F1_infected = {F1_infected_all.mean()}\n'
              f'Balanced accuracy = {balanced_accuracy_all.mean()}')

        # print(f'Std values:\n'
        #       f'Total accuracy = {total_acc_all.std()}\n'
        #       f'Accuracy_susceptible = {acc_susceptible_all.std()}\n'
        #       f'Accuracy_infected = {acc_infected_all.std()}\n'
        #       f'Accuracy_recovered = {acc_recovered_all.std()}\n'
        #       f'Precision_susceptible = {precision_susceptible_all.std()}\n'
        #       f'Precision_infected = {precision_infected_all.std()}\n'
        #       f'Precision_recovered = {precision_recovered_all.std()}\n'
        #       f'F1_infected = {F1_infected_all.std()}\n'
        #       f'balanced_accuracy = {balanced_accuracy_all.std()}')
        print('#####################################################################')
        # file_to_write.close()

    # print(model)

    # Improving the model
    # Cross-Validation
    # Hyperparameter Optimization
    # Different layer types GCN, GAT...
    # Different message passing layers
    # Including edge features
    # Featire scaling

    #networkx to visulize the graph
    #is addself loop required in forward?I think so it is done
    # GCN with skip connections
