from .._utils import *
from logparser.Drain import LogParser
from ast import literal_eval


class HDFS:
    def __init__(self):
        self.log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
        # Regular expression list for optional preprocessing (default: [])
        self.regex = [
            r'blk_(|-)[0-9]+' , # block id
            r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', # IP
            r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$', # Numbers
            ]
        self.st = 0.5  # Similarity threshold
        self.depth = 4  # Depth of all leaf nodes
        
    def get_csv(self, input_dir, output_dir, log_file):
        print('Start Parsing')
        parser = LogParser(self.log_format, indir=input_dir, outdir=output_dir,
                           depth=self.depth, st=self.st, rex=self.regex)
        parser.parse(log_file)
        print('Done')
        output_file = output_dir+log_file+'_structured.csv'
        return output_file
    
def get_BlockId(ParameterList):
    for s in ParameterList:
        sub = s.split(' ')
        # print(s)
        if 'blk' in sub[0][:3]:
            return sub[0]
    return None

def get_df(path:str, label_path:str):
    hdfs_df = pd.read_csv(path)
    
    print('literal_eval process...')
    hdfs_df['ParameterList'] = hdfs_df['ParameterList'].progress_apply(literal_eval)
    
    print('Getting BlockId process...')
    hdfs_df['BlockId'] = hdfs_df['ParameterList'].progress_apply(get_BlockId)
    hdfs_df = hdfs_df.dropna(subset='BlockId')
    label_df = pd.read_csv(label_path)
    
    print('Join the dataframe with labels')
    hdfs_df = hdfs_df.join(label_df.set_index('BlockId'), on='BlockId')
    
    hdfs_df = hdfs_df.reset_index(drop=True)
    
    print('Returning dataframe')
    return pd.DataFrame({'session_id':hdfs_df['BlockId'], 
                         'severity':hdfs_df['Level'],
                         'content':hdfs_df['Content'],
                         'event_id':hdfs_df['EventId'],
                         'event_template':hdfs_df['EventTemplate'],
                         'label':hdfs_df['Label']})

