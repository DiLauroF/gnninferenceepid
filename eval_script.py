import torch
from torch_geometric.data import Data, DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
random.seed(1)
from sklearn.metrics import confusion_matrix
from gnn_seird_dataloader import load_graphs, load_graphs_italy, balance_data, balance_data_italy,balance_data_lastdays
from gnn_seird_cluster import GCN_l,GCN, load_data, GAT
import tikzplotlib
import os


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
    return test_acc,cm,y_pred, y_true


# model = GCN_l(32, 0.3, 2)
num_features = 3
layers = 2
model = GCN_l(hidden_channels=64, dropout=0.4, no_layers=layers,num_features= num_features)
checkpoint = torch.load('./save_mode1_reduced_beta/model_scenario1_1')
model.load_state_dict(checkpoint['model_state_dict'])
scaler = checkpoint['scaler']
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
hidden_nodes = 0.8
print(f'No. of hidden nodes ={hidden_nodes}')

# weight_scale = 2
k_days = 15
no_graphs = 5
no_graphs_test = 50
file_loc_train = './graphs_boston'
file_loc_test = './graphs_1_beta_test'
plot_graph = 'False'

#Boston:300,scenario1:120/120
num_days = 120
start_day = 0
end_day = 10
#boston:100, scenario1:30,20
cuttoff_day =30

total_acc_all = np.array([])
acc_infected_all = np.array([])
acc_susceptible_all = np.array([])
acc_recovered_all = np.array([])
precision_susceptible_all = np.array([])
precision_infected_all = np.array([])
precision_recovered_all = np.array([])
F1_infected_all = np.array([])
balanced_accuracy_all = np.array([])



def cal_percentile(x, y, ax, color, n=20, alpha=0.25):
    # alpha= 1/n
    for percentile_min,percentile_max, alpha_color in zip([25,10],[75,90], [alpha,alpha*0.5]):
        perc1 = np.percentile(y, percentile_min, axis=0)
        perc2 = np.percentile(y, percentile_max, axis=0)
        ax.fill_between(x, perc1, perc2, alpha=alpha_color, color=color, edgecolor=None)

        # perc1 = np.percentile(y, np.linspace(percentile_min, 50, num=n, endpoint=False), axis=0)
        # perc2 = np.percentile(y, np.linspace(50, percentile_max, num=n+1)[1:], axis=0)
        # for p1, p2 in zip(perc1, perc2):
        #     ax.fill_between(x, p1, p2, alpha=alpha_color, color=color, edgecolor=None)


    return ax


def plot_predictions(df, y_pred,y_true,total_graphs,num_features):

    num_unknown_nodes = int(hidden_nodes*num_nodes)
    total_node_per_graph = int(len(df['id'])/total_graphs)
    total_unknown_node_per_graph = int(y_pred.shape[0]/total_graphs)



    for j in range(total_graphs):

        start_idx_node = j*total_node_per_graph
        end_idx_node = start_idx_node + total_node_per_graph
        start_idx_unknown = j*total_unknown_node_per_graph
        end_unknown_nodes = start_idx_unknown + total_unknown_node_per_graph
        unknown_idx = start_idx_unknown

        #Use-data for one of the days to identify the days to evaluate
        node_label = np.empty((0, num_days))
        node_usedata = np.empty((0, num_days))
        node_usedata = np.concatenate((node_usedata,np.array(df.use_data[np.arange(start_idx_node,end_idx_node,num_nodes)])[:, None].T))
        usedata_index = np.where(node_usedata==1)[1]
        node_label_pred = np.empty((0, usedata_index.shape[0]))

        if j ==0:
            day_labels_true = np.zeros([total_graphs,num_features,usedata_index.shape[0]])
            day_labels_pred = np.zeros([total_graphs,num_features,node_label_pred.shape[1]])

        for idx, day in enumerate(usedata_index):
            i = start_idx_node + day*num_nodes
            i_pred = start_idx_unknown + idx*num_unknown_nodes
            for nodestate in range(num_features):
                day_labels_true[j,nodestate,idx] = np.where(y_true[i_pred:i_pred+num_unknown_nodes] == nodestate)[0].shape[0]
                day_labels_pred[j,nodestate,idx] = np.where(y_pred[i_pred:i_pred+num_unknown_nodes] == nodestate)[0].shape[0]

    plot_save_dir = 'scenario1_h8'
    if not os.path.isdir(plot_save_dir):
        os.mkdir(plot_save_dir)
    for nodestate, graph_label in zip(range(num_features),['S','EI','RD']):
        fig, ax = plt.subplots()
        ax.plot(np.arange(usedata_index.shape[0]), np.median(day_labels_true[:,nodestate,:],axis=0),
                color='r',label=graph_label+' real')
        ax = cal_percentile(np.arange(usedata_index.shape[0]),day_labels_true[:,nodestate,:],
                            ax,color='r',n=10, alpha=0.15)

        ax.plot(np.arange(usedata_index.shape[0]), np.median(day_labels_pred[:,nodestate,:],axis=0),
                color='g',linestyle=':',label=graph_label+' pred')
        ax = cal_percentile(np.arange(usedata_index.shape[0]),day_labels_pred[:,nodestate,:],
                            ax, color='g', n=10, alpha=0.15)
        # ax[nodestate].fill_between(np.arange(usedata_index.shape[0]), mean_pred - std_pred, mean_pred + std_pred, color=color,alpha=0.1)
        ax.grid('on')
        ax.legend()
        plt.xlabel('time')
        plt.ylabel('N')
        plt.ylim(0,num_unknown_nodes)
        tikzplotlib.save(os.path.join(plot_save_dir,f"{graph_label}.tikz"))
        plt.savefig(os.path.join(plot_save_dir,f"{graph_label}.png"))
        plt.clf()


# for start_day,end_day in zip([0,15,35],[15,35,50]):
print(f'Cutoff_day={start_day}, end_day={end_day}, hidden_nodes={hidden_nodes}')

torch.cuda.empty_cache()
start_idx_graph = 0
df_test, num_nodes, num_days = load_graphs_italy(no_graphs_test, hidden_nodes,
                                                 file_loc_test,start_idx_graph, k_days,num_days)

# df_test = balance_data_lastdays(df_test, num_days, num_nodes, no_graphs_test, start_day,end_day)
# df_test = balance_data_italy(df_test, num_days, num_nodes, no_graphs_test, cuttoff_day)
known_idx_test = np.where(df_test['state'] == 1)[0]
df_test.values[known_idx_test,2:5] = scaler.transform(df_test.values[known_idx_test,2:5])
test_data_list = load_data(file_loc_test,df_test, no_graphs_test, num_days, num_nodes,device)
test_indices = [id for id in range(len(test_data_list))]
test_dataset = DataLoader(test_data_list, batch_size=1, sampler=test_indices)

test_acc, conf_matrix,y_pred, y_true = test(test_dataset)
print(f' Test Accuracy: {test_acc:.4f},\n'
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


plot_predictions(df_test, y_pred, y_true,no_graphs_test,num_features)

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


print('#####################################################################')