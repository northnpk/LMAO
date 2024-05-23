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

def get_embeddings(data, embeddings):
    a = np.zeros((len(data), embeddings.shape[1]))
    for indice in range(len(data)):
        if indice == -1 :
            continue
        else :
            a[indice] = embeddings[indice]
    return a

def get_freq(data, data_size):
    tmp = np.zeros(data.shape)
    for i in range(len(data)):
        tmp[i] = [data[i][0], data[i][1]/data_size]
    return tmp

def get_graph(seq, embeddings=None):
    edge_list=[]
    data_size = len(seq)
    for i in range(data_size):
        if i < data_size-1:
            edge_list.append((seq[i], seq[i+1]))
    c = Counter(seq)
    nodes = list(c.keys())
    # print(nodes)
    data = np.array(list(c.items()))
    x = get_freq(data, data_size)
        
    if embeddings != None:
        x = np.hstack([x, get_embeddings(data, embeddings)])
    else:
        x = data
        
    # print(x)
    node_list = []
    for node, features in zip(nodes, x):
        node_list.append((node, {a:float(b) for a, b in enumerate(features)}))
    # print(node_list)

    G = nx.DiGraph()
    G.add_nodes_from(node_list) # now the attributes are part of the original nx-graph
    G.add_edges_from(edge_list)
    return G

def get_PyG_data(row, group_node_attrs, embeddings=None):
    G = get_graph(row['X'], embeddings)
    data = from_networkx(G, group_node_attrs=group_node_attrs)
    data.y = row['y']
    return data

def getting_loader(df, group_node_attrs, embeddings=None, batch_size=32):
    return DataLoader(df.progress_apply(get_PyG_data, group_node_attrs=group_node_attrs,
                                        embeddings=embeddings, axis=1).to_list(), batch_size=batch_size)