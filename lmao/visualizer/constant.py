from .._utils import *

def get_constants(df):
    total_log = len(df)
    total_sessions = len(df.groupby('session_id'))
    total_topics = len(df['topic'].unique())
    label_counts = df['label'].value_counts()
    normal_anomaly_ratio = f'{label_counts[0]/label_counts[1]:.2f}:1'
    return total_log,total_sessions,total_topics,normal_anomaly_ratio