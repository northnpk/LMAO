from tqdm import tqdm
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt

def get_node(df: pd.DataFrame,
             node_cols: str = 'EventId',
             severity_cols: str = 'Severity', 
             return_colormaps: bool = True, 
             window_size: int = 1000,
             severity_color_map: dict = {'Start':'green', 'E':'red', 'Else':'grey'}):
    Event = df[node_cols].to_list()
    Severity = df[severity_cols].to_list()
    node1 = []
    node2 = []
    color_map = []
    print('Getting nodes...')
    for i in tqdm(range(len(Event)), total=len(Event)):
        rest = len(Event)-i
        if rest <= window_size:
            for j in range(i+1, rest):
                node1.append(Event[i])
                node2.append(Event[j])
        else : 
            for j in range(i+1, i+window_size):
                node1.append(Event[i])
                node2.append(Event[j])
        if return_colormaps == True:
            if i == 0:
                color = severity_color_map['Start']
            else :
                if Severity[i] in severity_color_map.keys:
                    color = severity_color_map[Severity[i]]
                else : 
                    color = 'grey'
                    
            node_color = {'node':Event[i], 'color': color}
            if node_color not in color_map:
                color_map.append(node_color)
        
    print('Creating dataframe...')
    if return_colormaps == True:
        return pd.DataFrame({'node1':node1, 'node2':node2}), pd.DataFrame(color_map)
    else:
        return pd.DataFrame({'node1':node1, 'node2':node2}), None
    

def get_networkx(df_node: pd.DataFrame):
    
    print('Creating graph...')
    # Build your graph
    G = nx.from_pandas_edgelist(df_node, 'node1', 'node2')
    return G

def plot_run(G,
              min_edge: bool=True,
              color_map: pd.DataFrame=None,
              **kwds):
    
    if color_map != None:
        print('creating color map...')
        color_map.drop_duplicates(subset=['node'], inplace=True)
        color_map.set_index('node')
        color_map.reindex(G.nodes())
        print('Done')
    
    # fixing the size of the figure
    plt.figure(figsize =(10, 7))
    if min_edge:
        nx.draw_networkx(G, with_labels = True, arrows=True, edgelist=nx.min_edge_cover(G), node_color=color_map['color'], kwds=kwds)
    else: 
        nx.draw_networkx(G, with_labels = True, arrows=True, node_color=color_map['color'], kwds=kwds)
    plt.axis('off')
    plt.show()
    

def plot_each_run(df:pd.DataFrame,
                  min_edge:bool=False,
                  **kwds):
    run_id = df['RunID'].unique()
    print('Done')
    for id in run_id:
        print('Getting nodes for run:', id)
        df_node, color_map = get_node(df[df['RunID'] == id])
        G = get_networkx(df_node=df_node)
        print('Plotting run:', id)
        # fixing the size of the figure
        plot_run(G=G, min_edge=min_edge, color_map=color_map, kwds=kwds)
    print('Done')