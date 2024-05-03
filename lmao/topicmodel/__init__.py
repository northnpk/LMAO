from .._utils import *
from ..topicmodel import bert

class LMAOTopic:
    def __init__(self, mode='BERTopic'):
        self.model = None
        self.mode = mode
        self.df = pd.DataFrame()
        print(f'Topic model mode set to :{self.mode}.')
        if self.mode == 'BERTopic':
            print(f'Start Initialized')
        else :
            print(f'We do not have the {mode} mode yet.')
    
    def training_model(self, df:pd.DataFrame):
        if self.mode == 'BERTopic':
            embedding_path = bert.save_embeddings(df, batch_size=512)
            self.model = bert.update_model(df, data_col='content', embeddings_load=embedding_path)
        else :
            print(f'We do not have the {self.mode} mode yet.')
    
    def get_topic(self, df:pd.DataFrame, map:dict=None):
        if map :
            df['topic'] = df['topic'].progress_apply(lambda x: map(x))
        elif self.mode == 'BERTopic':
            df['topic'] = self.model.topics_
        else :
            print(f'We do not have the {self.mode} mode yet.')
        return df