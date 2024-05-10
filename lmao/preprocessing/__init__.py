from .._utils import *
from ..preprocessing import hdfs, alice_info

class LMAOPrep:
    def __init__(self, data_type='hdfs', parsing=True):
        super().__init__()
        self.data_type = data_type
        self.parsing = parsing
        self.csv_path = None
        self.df = pd.DataFrame()
        print('Created prep class')
        
    def log_parsing(self, input_dir, log_file, output_dir):
        if self.data_type == 'hdfs':
            self.csv_path = hdfs.HDFS().get_csv(input_dir=input_dir,
                                                log_file=log_file,
                                                output_dir=output_dir)
        else :
            print(f'This version not having log_parsing for {self.data_type} type yet.')
            pass
        return self.csv_path
    
    def from_parsed_csv(self, csv_path):
        self.csv_path = csv_path
        return self.csv_path
        
    def prep_data(self, label_path:str, eor_path:str=None):
        print(f'Preprocessing to LMAO format with label from {self.data_type}')
        print(f'Path :{label_path}')
        if self.data_type == 'hdfs':
            self.df = hdfs.get_df(self.csv_path, label_path)
        elif self.data_type == 'alice_info':
            self.df = alice_info.get_df(self.csv_path, label_path, eor_path)
        else :
            print(f'This version not having prep_data for {self.data_type} type yet.')
            pass
        print('Done')
    
    def load_csv(self, csv_path:str):
        print(f'DataFrame from csv path:{csv_path}')
        self.df = pd.read_csv(csv_path)
        return self.df
    
    def load_parquet(self, parquet_path:str):
        print(f'DataFrame from parquet path:{parquet_path}')
        self.df = pd.read_parquet(parquet_path)
        return self.df