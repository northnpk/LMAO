from .._utils import *

crashed = ['request from flp expert.',
           'ali-ecs core restarted',
           'mch-qcmn-epn-full-track-matching crashed.',
           'triggers out synch. disabling ctp readout.',
           'trigger readout went to zero.',
           'end calibration',
           'end test',
           'change luminosity',
           'tpc scan',
           'rm asks to stop the run',
           'sor call failed',
           'operator error: run restarted the same partition.',
           'need to restart trigger.',
           ]

def get_run_id(df:pd.DataFrame, run_df:pd.DataFrame):
    df['RunID'] = 0
    run_df.dropna(subset=['time_o2_start'], inplace=True)
    run_df.dropna(subset=['time_o2_end'], inplace=True)
    time_start = run_df['time_o2_start']
    time_end = run_df['time_o2_end']
    run_id = run_df['id']
    run_quality = run_df['run_quality']
    length = len(run_df)
    for idx in tqdm(range(length), total=length):
        start_date = time_start.iloc[idx]
        end_date = time_end.iloc[idx]
        id = run_id.iloc[idx]
        quality = run_quality.iloc[idx]
        if type(start_date) == float and type(end_date) == float:
            continue
        else : 
            df.loc[(df['date'] >= start_date) & (df['date'] <= end_date),'RunID'] = id
            df.loc[(df['date'] >= start_date) & (df['date'] <= end_date),'RunQuality'] = quality
        # print(run_id, start_date, type(end_date), run_quality)
    df = df[df['RunID'] != 0]
    return df

def get_eor(df:pd.DataFrame, eor_df:pd.DataFrame):
    df['EOR'] = None
    eor_id = eor_df['run_id']
    description = eor_df['description']
    reason_type_id	= eor_df['reason_type_id']
    for idx in tqdm(range(len(eor_id)), total=len(eor_id)):
        des = description.iloc[idx]
        r_id = reason_type_id.iloc[idx]
        id = eor_id.iloc[idx]
        df.loc[(df['RunID'] == id) ,'EOR'] = des
        df.loc[(df['RunID'] == id) ,'EORTypeID'] = r_id
    return df[df['EOR']!=None]

def normalize(text):
    if type(text) == str:
        text = text.lower()
        text = text.replace('of ', '')
        text = text.replace('in ', '')
        return text
    else :
        return text
        
def get_df(csv_path:str, label_path:str, eor_path:str):
    df = pd.read_csv(csv_path)
    run_df = pd.read_csv(label_path)
    eor_df = pd.read_csv(eor_path)
    df['date'] = pd.to_datetime(df['Timestamp'], unit='s')
    
    print('Stating map RunID from label file')
    df = get_run_id(df, run_df)
    
    print('Stating map EOR from EOR file')
    df = get_eor(df, eor_df)
    
    print('Drop all severity D')
    df = df[df['Severity']!='D']
    
    print('Normalize EOR')
    df['EOR'] = df['EOR'].progress_apply(normalize)
    df = df.dropna(subset='EOR')
    
    print('Getting Crashed from EOR')
    df['crash'] = ['crashed' if eor in crashed else 'not crashed' for eor in df['EOR'].to_list()]
    
    df = df.reset_index(drop=True)
    
    print('Returning dataframe')
    return pd.DataFrame({'session_id':df['RunID'],
                         'severity':df['Severity'],
                         'content':df['Content'],
                         'event_id':df['EventId'],
                         'event_template':df['EventTemplate'],
                         'label':df['crash'],
                         'EOR':df['EOR'],
                         'EOR_id':df['EORTypeID'],
                         'run_quality':df['RunQuality']})
    # return df