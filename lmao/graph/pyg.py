from .._utils import *
from collections import Counter
import networkx as nx

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx

class2bin = {'Normal':0,
             'Anomaly':1}
bin2class = ['Normal', 'Anomaly']
def class_trans(data, mode=class2bin):
    return mode[data]

class PyGraph:
    def __init__(self, df:pd.DataFrame, mode='topic_seq', seq_len:int=300):
        df['y'] = df['y'].apply(class_trans, mode=class2bin)
        if mode == 'topic_only':
            self.n_features = 2
            self.group_node_attrs = list(map(lambda x: str(x), range(0, self.n_features)))
        elif mode == 'topic_seq':
            self.n_features = seq_len+2
            self.group_node_attrs = list(map(lambda x: str(x), range(0, self.n_features)))
        
        else :
            print(f'We do not have the {mode} mode yet. Using topic_seq mode.')
            self.n_features = seq_len+2
            self.group_node_attrs = list(map(lambda x: str(x), range(0, self.n_features)))
        
        return 
        

def pos_onehot(seq, keys):
    a = np.zeros((len(keys), 300))
    for i, n in enumerate(keys):
        pos = np.where(seq==n)[0]
        for p in pos:
            a[i][p] = 1
    return a

def get_graph(seq):
    edge_list=[]
    data_size = len(seq)
    # for idx, node in tqdm(enumerate(seq), total=data_size):
    for i in range(data_size):
        if i < data_size-1:
            edge_list.append((seq[i], seq[i+1]))
    c = Counter(seq)
    nodes = list(c.keys())
    # print(nodes)
    data = np.array(list(c.items()))
    pos = pos_onehot(seq, nodes)
    x = np.hstack([data, pos])
    # print(x)
    node_list = []
    for node, features in zip(nodes, x):
        node_list.append((node, {a:float(b) for a, b in enumerate(features)}))
    # print(node_list)

    G = nx.DiGraph()
    G.add_nodes_from(node_list) # now the attributes are part of the original nx-graph
    G.add_edges_from(edge_list)
    return G

def get_PyG_data(row, group_node_attrs):
    G = get_graph(row['X'])
    data = from_networkx(G, group_node_attrs=group_node_attrs)
    data.y = row['y']
    return data

def getting_loader(df, group_node_attrs, batch_size=32):
    return DataLoader(df.progress_apply(get_PyG_data, group_node_attrs, axis=1).to_list(), batch_size=batch_size)