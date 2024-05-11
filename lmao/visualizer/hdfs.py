from .._utils import *
from lmao.topicmodel import LMAOTopic
from lmao.graph import LMAOGraph
from lmao.visualizer.plot import plotly_digraph

topic_model = LMAOTopic()
topic_model.from_trained_model(model_path='lmao/topicmodel/model/BERTopic-HDFS-all-MiniLM-L6-v2')

df = pd.read_parquet('dataset/HDFS_v1/HDFS_with_topic.parquet.gzip')
data = LMAOGraph(df=df,mode='pyg', padding=False)

fig1 = topic_model.model.visualize_topics(top_n_topics=20)
fig2 = topic_model.model.visualize_heatmap(top_n_topics=20)
fig2.update_xaxes(visible=False)
fig2.update_yaxes(visible=False)
topics_per_class = topic_model.model.topics_per_class(df.content.tolist(), df.label.tolist())
fig3 = topic_model.model.visualize_topics_per_class(topics_per_class)

def generate_visualizations():
    return fig1, fig2, fig3

def get_constants():
    total_log = len(df)
    total_sessions = len(df.groupby('session_id'))
    total_topics = len(df['topic'].unique())
    label_counts = df['label'].value_counts()
    normal_anomaly_ratio = f'{label_counts[0]/label_counts[1]:.2f}:1'
    return total_log,total_sessions,total_topics,normal_anomaly_ratio

def get_graph(i=0):
    G = data.get_one_graph(i, feature=False)
    return plotly_digraph(G, label=data.df['y'][i], topic_dict=topic_model.topic_dict)