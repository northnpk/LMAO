from .._utils import *
from ..preprocessing import hdfs, alice_info

class LMAOPreprocessing:
    def __init__(self, data_type='hdfs', parsing=True):
        super(LMAOPreprocessing, self).__init__()
        self.data_type = data_type
        self.parsing = parsing
        return print(self)
        
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
        
    def prep_data(self, label_path):
        print(f'Preprocessing to LMAO format with label from {self.data_type}')
        print(f'Path :{label_path}')
        if self.data_type == 'hdfs':
            self.df = hdfs.get_df(self.csv_path, label_path)
        if self.data_type == 'alice':
            self.df = alice_info.get_df()
        else :
            print(f'This version not having prep_data for {self.data_type} type yet.')
            pass
        print('Done')
        
    def to_csv(self, file_name:str, *args):
        print('Saving to CSV file')
        self.df.to_csv(file_name, args)
        print('done')
    
    def to_parquet(self, file_name:str, *args):
        print('Saving to CSV file')
        self.df.to_parquet(f'{file_name}.parquet.gzip',compression='gzip')
    
    def load_csv(self, csv_path:str):
        print(f'DataFrame from csv path:{csv_path}')
        self.df = pd.read_csv(csv_path)
        return self.df
    
    def load_parquet(self, parquet_path:str):
        print(f'DataFrame from parquet path:{parquet_path}')
        self.df = pd.read_parquet(parquet_path)