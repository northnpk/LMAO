from .._utils import *
from ..topicmodel import bert


class LMAOTopic:
    def __init__(self, mode='BERTopic'):
        self.model = None
        self.mode = mode
        self.df = pd.DataFrame()
        self.topic_dict = None
        self.topic_pos = None
        print(f'Topic model mode set to :{self.mode}.')
        if self.mode == 'BERTopic':
            print(f'Start Initialized')
        else :
            print(f'We do not have the {mode} mode yet.')
    
    def training_model(self, df:pd.DataFrame, embedding_path:str=None):
        if self.mode == 'BERTopic':
            if embedding_path:
                embedding_path = bert.save_embeddings(df, batch_size=512)
            self.model = bert.update_model(df, data_col='content', embeddings_load=embedding_path)
            self.topic_dict = self.get_topic_dict()
            # self.topic_pos = bert.get_coor_topic(self)
        else :
            print(f'We do not have the {self.mode} mode yet.')
    
    def save_model(self, model_path:str):
        if self.mode == 'BERTopic':
            bert.save_model(self.model, path_to_save=model_path, embedding_model="sentence-transformers/all-MiniLM-L6-v2")
        else :
            print(f'We do not have the {self.mode} mode yet.')
    
    def from_trained_model(self, model_path:str, embedding_model_name:str="all-MiniLM-L6-v2"):
        if self.mode == 'BERTopic':
            self.model = bert.load_model(path_to_load=model_path, embedding_model_name=embedding_model_name)
            self.topic_dict = self.get_topic_dict()
            # self.topic_pos = bert.get_coor_topic(self)
            print('Done')
        else :
            print(f'We do not have the {self.mode} mode yet.')
    
    def get_topic(self, df:pd.DataFrame, map:dict=None):
        if map :
            df['topic'] = df['content'].progress_apply(lambda x: map(x))
        elif self.mode == 'BERTopic':
            df['topic'] = self.model.topics_
        else :
            print(f'We do not have the {self.mode} mode yet.')
        return df
    
    def get_topic_dict(self):
        if self.mode == 'BERTopic':
            topics_name_dict = self.model.get_topic_info().set_index('Topic')['Name'].T.to_dict()
            topics_name_dict[-2] = '<PAD>'
            return topics_name_dict
        else :
            print(f'We do not have the {self.mode} mode yet.')
            return None
        
        