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

def pos_onehot(seq, keys, len_seq):
    a = np.zeros((len(keys), len_seq))
    for i, n in enumerate(keys):
        pos = np.where(seq==n)[0]
        for p in pos:
            a[i][p] = 1
    return a

def get_graph(seq, len_seq:int=300):
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
    pos = pos_onehot(seq, nodes, len_seq)
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

def get_PyG_data(row, group_node_attrs, len_seq:int=300):
    G = get_graph(row['X'], len_seq=len_seq)
    data = from_networkx(G, group_node_attrs=group_node_attrs)
    data.y = row['y']
    return data

def getting_loader(df, group_node_attrs, batch_size=32, len_seq:int=300):
    return DataLoader(df.progress_apply(get_PyG_data, group_node_attrs=group_node_attrs,
                                        len_seq=len_seq, axis=1).to_list(), batch_size=batch_size)