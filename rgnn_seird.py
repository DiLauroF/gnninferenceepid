# -*- coding: utf-8 -*-
"""
@author: Abhishek
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
from torch_geometric import utils
from torch_geometric.nn import GCNConv,SAGEConv



from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix


from gnn_seird_dataloader import load_graphs, balance_data
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

torch.cuda.empty_cache()

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def GNN_block(in_f, out_f, dropout, **kwargs):
    return nn.Sequential(
        GCNConv(in_f, out_f),
        nn.ReLU(),
        #distinction b/w training and eval
        nn.Dropout(dropout)
    )

######## CLASS
#Method 1 from paper:STRUCTURED SEQUENCE MODELING WITH GRAPH CONVOLUTIONAL RECURRENT NETWORKS

class Recurrent_GCN_1(torch.nn.Module):
    def __init__(self, hidden_channels,dropout):
        # Init parent
        super(Recurrent_GCN_1, self).__init__()
        torch.manual_seed(42)
        self.dropout = dropout
        self.hidden_channels = hidden_channels

        # GCN layers: 2 message passing layers (to create embedding)
        self.conv1 = GCNConv(num_features, self.hidden_channels)
        self.conv2 = GCNConv(self.hidden_channels, self.hidden_channels)

        self.layer_count = 1
        self.hidden_c = nn.Parameter(torch.randn(self.layer_count, 1, self.hidden_channels).cuda(), requires_grad=True)
        self.Recurrent_unit1 = nn.GRUCell(self.hidden_channels, self.hidden_channels)

        # Output layer: is the classification layer
        self.out = Linear(hidden_channels, 5)

    def forward(self, _input):
        x, edge_index, batch = _input.x, _input.edge_index, _input.batch
        for k_lookback in range(lookback):
            if k_lookback == 0:
                next_hidden = self.init_hidden(batch_size=x.shape[0])[0]

            x_gnn = self.conv1(x[:,k_lookback,:], edge_index)
            x_gnn = x_gnn.relu() # the classical activation function
            x_gnn = F.dropout(x_gnn, p=self.dropout, training=self.training)# and dropout to avoid overfitting

            # Second Message Passing layer
            x_gnn = self.conv2(x_gnn, edge_index)
            x_gnn = x_gnn.relu()
            x_gnn = F.dropout(x_gnn, p=self.dropout, training=self.training)


            #RNN
            next_hidden = self.Recurrent_unit1(x_gnn, next_hidden)

        # Output layer: as in NN output activation function with a probability (to be a certain class) as ouput
        x_gnn = F.softmax(self.out(next_hidden), dim=1) #is the classification layer

        return x_gnn

    # def init_hidden(self, batch_size=1):
    #     return self.hidden_c.expand(self.layer_count, batch_size, self.hidden_channels).contiguous()

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.layer_count, batch_size, self.hidden_channels).zero_().to(device)
        return hidden

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels,dropout):
        # Init parent
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.dropout = dropout

        # GCN layers: 2 message passing layers (to create embedding)
        # self.conv1 = GCNConv(num_features, hidden_channels)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv = nn.Sequential(
                    GNN_block(num_features, hidden_channels),
                    GNN_block(hidden_channels, hidden_channels)
                    )

        # Output layer: is the classification layer
        self.out = Linear(hidden_channels, 5)

    def forward(self, _input):
        x_in, edge_index, batch = _input.x, _input.edge_index, _input.batch
        # First Message Passing layer: is equal as in NN with the edge info
        x = self.conv(x_in[:,-1,:], edge_index)
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

def load_data_rnn(path, data_frame, lookback,total_graphs):
    inputs = np.zeros((len(data_frame)-total_graphs*(lookback-1)*num_nodes,lookback,num_features))
    labels = np.zeros(len(data_frame)-total_graphs*(lookback-1)*num_nodes)
    state_node = np.zeros(len(data_frame)-total_graphs*(lookback-1)*num_nodes)
    use_data = np.zeros(len(data_frame)-total_graphs*(lookback-1)*num_nodes)

    #use df.t
    k_input = 0
    for j in range(total_graphs):
        for i in range(lookback, int(num_days+1)):
            start_idx = (i-lookback)*num_nodes + j*(num_nodes*num_days)
            end_idx = (i*num_nodes + j*(num_nodes*num_days))
            temp_mat = data_frame.values[start_idx:end_idx,2:7]
            temp_mat = temp_mat.reshape(lookback,num_nodes,-1)
            temp_mat = np.moveaxis(temp_mat,[0,1,2],[1,0,2])

            inputs[k_input:k_input+num_nodes] = temp_mat
            labels[k_input:k_input+num_nodes] = data_frame.values[end_idx-num_nodes:end_idx,7]
            state_node[k_input:k_input+num_nodes] = data_frame.values[end_idx-num_nodes:end_idx,8]
            use_data[k_input:k_input+num_nodes] = data_frame.values[end_idx-num_nodes:end_idx,9]
            k_input += num_nodes

    x = torch.tensor(inputs, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    data_list = []
    for j in range(total_graphs):
        loc = os.path.join(path,f'Adjacency_matrix_edgelist_{j}.csv')
        edges = pd.read_csv(loc, header=None, sep=';').to_numpy()
        # Edges list: in the format (head, tail); the order is irrelevant
        edge_index = utils.to_undirected(torch.tensor(edges, dtype=torch.long).t())
        for i in range(num_days-lookback+1):
            start_idx = i*num_nodes + j*(num_nodes*(num_days-lookback+1))
            end_idx = i*num_nodes + num_nodes + j*(num_nodes*(num_days - lookback+1))
            if use_data[start_idx] == 1:
                data_list.append(Data(x=x[start_idx:end_idx,:,:], y=y[start_idx:end_idx],
                                  edge_index=edge_index.to(device), state=state_node[start_idx:end_idx],
                                  graph_idx = loc))

    return data_list
######## PREPROCESS DATA

class GNNDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GNNDataset, self).__init__(root,transform, pre_transform)
        self.df = df
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
        path = './graphs'
        data_list = []
        x = torch.tensor(df.values[:,2:-2], dtype=torch.float)
        y = torch.tensor(df['y'], dtype=torch.long)
        for j in range(no_graphs):
            loc = os.path.join(path,f'Adjacency_matrix_edgelist_{j}.csv')
            edges = pd.read_csv(loc, header=None, sep=';').to_numpy()
            # Edges list: in the format (head, tail); the order is irrelevant
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            for i in range(num_days):
                start_idx = i*num_nodes + j*num_nodes*num_days
                end_idx = i*num_nodes+ j*num_nodes*num_days + num_nodes
                data_list.append(Data(x=x[start_idx:end_idx,:], y=y[start_idx:end_idx],
                                      edge_index=edge_index, state=df.values[start_idx:end_idx,-1]))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# Load data
no_graphs = 80
no_graphs_test = 20
hidden_nodes = 0.2
k_days = 15
dropout_hidden = 0.3
weight_scale = 2
print(f'Evaluation: run rgnn on 80 realisation')
print(f'Train graphs:{no_graphs},Test graphs:{no_graphs_test},'
      f'hidden_nodes:{hidden_nodes}, k_days:{k_days}')
file_loc_train = './graphs'
file_loc_test = './graphs_test_1'
# num_nodes = 1000
# num_days =365
start_idx_graph = 0
df, num_nodes, num_days = load_graphs(no_graphs, hidden_nodes, file_loc_train, start_idx_graph,k_days)
print(f'No.of_days_used:{num_days}')
df = balance_data(df, num_days, num_nodes, no_graphs)
# feature scaling
sc = MinMaxScaler()
known_idx_train = np.where(df['state'] == 1)[0]
scaler = sc.fit(df.values[known_idx_train,2:7])
df.values[known_idx_train,2:7] = scaler.transform(df.values[known_idx_train,2:7])

# Define lookback period and split inputs/labels
lookback = 7


# use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# train_data_list = GNNDataset('./')
# train_data_list = train_data_list.shuffle()
# train_dataset = DataLoader(train_data_list, batch_size=512)

num_features = 5
train_data_list = load_data_rnn(file_loc_train, df, lookback, no_graphs)
train_indices = [id for id in range(len(train_data_list))]
random.shuffle(train_indices)
train_dataset = DataLoader(train_data_list, batch_size=512, sampler=train_indices)


# train_dataset.num_classes = 5 # S,E,I,R,D

# Initialize Model
model = Recurrent_GCN_1(hidden_channels=32, dropout=dropout_hidden)
print(model)
model = model.to(device)


learning_rate = 0.0005 # step for gradient descendent method for learning (?)
decay = 5e-4 #decay of importance of learning rate
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)

# Define loss function (CrossEntropyLoss for Classification problems with probability distribution)
w0 = np.where(df['y'] == 0)[0].shape[0]
w1 = np.where(df['y'] == 1)[0].shape[0]
w2 = np.where(df['y'] == 2)[0].shape[0]
w3 = np.where(df['y'] == 3)[0].shape[0]
w4 = np.where(df['y'] == 4)[0].shape[0]
max_w = max(w0,w1,w2,w3,w4)
weight = torch.tensor([max_w/w0,(max_w/w1),(max_w/w2)*weight_scale,(max_w/w3),(max_w/w4)]).to(device)
print(weight)
criterion = torch.nn.CrossEntropyLoss(weight=weight)
# cross entropy compare probabilities, and we have probabilities because of softmax

# Train !
del df
del train_data_list
losses = []
start = time.time()
for epoch in range(501):
    loss = train(train_dataset)
    losses.append(loss)
    if epoch % 100 == 0:
        # train_acc,_ = test(train_dataset)
        # test_acc,_ = test(test_dataset)
        print(f'[{time_since(start)}], Epoch: {epoch:03d}, Loss: {loss:.4f}')

# Train results
# import seaborn as sns
# losses_float = [float(loss.cpu().detach().numpy()) for loss in losses]
# loss_indices = [i for i,l in enumerate(losses_float)]
# sns.lineplot(loss_indices, losses_float)
# plt.show()


######################################################
# Test the model with different unknown nodes

del train_dataset

for n_test in range(3):
    torch.cuda.empty_cache()
    start_idx_graph = n_test * no_graphs_test
    df_test, _, _ = load_graphs(no_graphs_test, hidden_nodes, file_loc_test,start_idx_graph, k_days)
    known_idx_test = np.where(df_test['state'] == 1)[0]
    df_test.values[known_idx_test,2:7] = scaler.transform(df_test.values[known_idx_test,2:7])
    test_data_list = load_data_rnn(file_loc_test,df_test, lookback, no_graphs_test)
    test_indices = [id for id in range(len(test_data_list))]
    test_dataset = DataLoader(test_data_list, batch_size=256, sampler=test_indices)
    test_acc, conf_matrix = test(test_dataset)
    print(f'Test_set:{n_test}, Test Accuracy: {test_acc:.4f},\n'
          f' confusion_matrix:{conf_matrix}')
    infected_acc = conf_matrix[2,2]/np.sum(conf_matrix[2,:])
    infected_precision = conf_matrix[2,2]/np.sum(conf_matrix[:,2])
    F1_infected = 2* (infected_acc*infected_precision)/(infected_acc+infected_precision)
    print(f'Accuracy_infected = {infected_acc}\n'
          f'Precision_infected = {infected_precision}\n'
          f'F1_score_infected = {F1_infected}')

    del df_test
    del test_data_list
    del test_indices
    del test_dataset

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
