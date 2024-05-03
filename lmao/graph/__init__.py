from .._utils import *
import networkx as nx
import plotly.express as px
class graph:
    def __init__(self, df:pd.DataFrame,
                 group_name:str='session_id',
                 X_col:str='topic', y_col:str='label',
                 mode='seq', padding=True, seq_size:int=300):
        self.mode = mode
        print('Getting Group by from dataframe')
        self.df = group_to_classify(df=df, group_name=group_name,
                                    X_col=X_col, y_col=y_col)
        if padding :
            print(f'Apply Padding and Truncate {seq_size}.')
            self.df['X'] = self.df['X'].progress_apply(pad_truncate, seq_size)
        
        if mode == 'seq':
            return self.df
        elif mode == 'pyg':
            return self.df
        else :
            print(f'We do not have the {mode} mode yet.')
            return self.df
        
    
def group_to_classify(df:pd.DataFrame,
                      group_name:str='session_id',
                      X_col:str='topic', y_col:str='label'):
    X = []
    y = []
    for group_name, df_group in tqdm(df.groupby(group_name)):
        sub_X = df_group[X_col].to_list()
        sub_y = df_group[y_col].iloc[0]
        # len_X = len(sub_X)
        # print(f'len X :{len_X}')
        X.append(sub_X)
        y.append(sub_y)
    return pd.DataFrame({'X':X, 'y':y})

def pad_truncate(seq, max_len:int=300):
    if len(seq) >= max_len:
        return seq[:max_len]
    else:
        pad_length = max_len - len(seq)
        padded_sequence = np.pad(seq, (0, pad_length), mode='constant', constant_values=-2)
        return padded_sequence