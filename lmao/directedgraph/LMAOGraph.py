from ._utils import *

class LMAOGraph():
    def __init__(self, df: pd.DataFrame,
             node_cols: str = 'EventId',
             severity_cols: str = 'Severity', 
             return_colormaps: bool = True, 
             window_size: int = 1000,
             severity_color_map: dict = {'Start':'green', 'E':'red', 'Else':'grey'}
             ):
        self.node_df, self.color_map = get_node(df=df,
                                node_cols=node_cols,
                                severity_cols=severity_cols,
                                return_colormaps=return_colormaps,
                                window_size=window_size,
                                severity_color_map=severity_color_map)
        print('created')
        return self.node_df
            
    def get_node(self, df: pd.DataFrame,
             node_cols: str = 'EventId',
             severity_cols: str = 'Severity', 
             return_colormaps: bool = True, 
             window_size: int = 1000,
             severity_color_map: dict = {'Start':'green', 'E':'red', 'Else':'grey'}
             ):
        
        self.severity_color_map = severity_color_map
        
        self.node_df, self.color_map = get_node(df=df,
                                node_cols=node_cols,
                                severity_cols=severity_cols,
                                return_colormaps=return_colormaps,
                                window_size=window_size, 
                                severity_color_map=severity_color_map)
        
        return self.node_df
    
    def set_label(self, label:str='Normal'):
        self.label = label
        print(f'Graph set label to {label}.')
        
    def to_df(self):
        return pd.DataFrame({'Graph':self.node_df, 'Label':self.label})