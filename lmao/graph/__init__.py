from .._utils import *
import networkx as nx
import plotly.express as px
from collections import Counter
from .pyg import getting_loader, get_graph

class Graph:
    def __init__(self, df:pd.DataFrame,
                 group_name:str='session_id',
                 X_col:str='topic', y_col:str='label',
                 mode:str='seq', padding:bool=True, seq_size:int=300,
                 including_ref:bool=False) -> pd.DataFrame:
        self.graph_list = []
        self.mode = mode
        self.n_topics = len(df['topic'].unique())
        self.seq_size = None
        print(f'Topics size of this dataframe is : {self.n_topics}')
        if self.mode in ['seq','pyg']:
            print('Getting Group by from dataframe')
            self.df = group_to_classify(df=df, group_name=group_name,
                                    X_col=X_col, y_col=y_col)
            self.seq_size = self.df['X'].apply(len).max()
            print(f'Max sequence size of this dataframe is : {self.seq_size}')

            if padding :
                print(f'Apply Padding and Truncate {seq_size}.')
                self.df['X'] = self.df['X'].progress_apply(pad_truncate, max_len=seq_size)
                self.seq_size = seq_size
                self.n_topics = len(df['topic'].unique()) + 1 # Adding Pad (-2) to the topic input
                print(f'Updating Topic size of this dataframe to {self.n_topics}.')
                print(f'Updating sequence size of this dataframe to {self.seq_size}.')

            if self.mode == 'pyg':
                print('Please call .get_PyG() to get PyG dataloader.')
        
        elif self.mode == 'freq':
            print('Getting Group by from dataframe')
            self.df = group_to_classify(df=df, group_name=group_name,
                                    X_col=X_col, y_col=y_col)
            
            print('Apply Count to calculate each topic frequency')
            self.df['X'] = self.df['X'].progress_apply(get_freq, n_topics=self.n_topics)
            print(f'Expanding X into {self.n_topics} features')
            self.df = pd.concat([self.df['X'].progress_apply(pd.Series), self.df['y']], axis=1)
            self.df.rename(columns={self.n_topics-1:-1}, inplace = True)
                        
        else :
            print(f'We do not have the {mode} mode yet.')
            self.df = df
            
    def get_networkx_list(self, one_hot:bool=False):
        n_features = 2
        if one_hot:
            n_features+=self.seq_size
        print('Getting networkx Graph from dataframe')
        self.graph_list = self.df['X'].progress_apply(get_graph)
        return self.graph_list
    
    def get_PyG(self, one_hot:bool=False, batch_size:int=32, embeddings_path=None):
        n_features = 2
        if one_hot:
            n_features+=self.seq_size
        group_node_attrs = list(map(lambda x: str(x), range(0, n_features)))
        print('Getting PyG Loader Graph from dataframe')
        return getting_loader(df=self.df, group_node_attrs=group_node_attrs,
                              batch_size=batch_size, len_seq=self.seq_size)
        
    def get_one_graph(self, i, feature:bool=False):
        return get_graph(seq=self.df['X'][i], len_seq=self.seq_size, feature=feature)
        
    
def group_to_classify(df:pd.DataFrame,
                      group_name:str='session_id',
                      X_col:str='topic', y_col:str='label'):
    X = []
    y = []
    # log_idx = []
    for group_name, df_group in tqdm(df.groupby(group_name)):
        sub_X = df_group[X_col].to_list()
        sub_y = df_group[y_col].iloc[0]
        sub_log_idx = df_group.index.to_list()
        # len_X = len(sub_X)
        # print(f'len X :{len_X}')
        X.append(sub_X)
        y.append(sub_y)
        # log_idx.append(sub_log_idx)
    
    return pd.DataFrame({'X':X, 'y':y})

def pad_truncate(seq, max_len:int=300):
    if len(seq) >= max_len:
        return seq[:max_len]
    else:
        pad_length = max_len - len(seq)
        padded_sequence = np.pad(seq, (0, pad_length), mode='constant', constant_values=-2)
        return padded_sequence
    
def get_freq(seq, n_topics:int=2200): #topic from Alice_infologger is 2148 topics
    topics_key = np.zeros((n_topics))
    # print(topics_key.shape)
    c = Counter(seq)
    len_seq = len(seq)
    for a, b in c.items():
        topics_key[a] = b / len_seq
        # print(f'{a}:{b}')
    return topics_key