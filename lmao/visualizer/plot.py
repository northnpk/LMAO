from .._utils import *
import plotly.graph_objects as go
import networkx as nx

def normalized(a):
    return [float(i)/max(a) for i in a]

def plotly_digraph(G, label:str='Label', topic_dict:dict=None):
    
    # node_name = [f'Topic:{int(G.nodes[n][0])}, Name:{topic_dict[int(G.nodes[n][0])]}, Count:{int(G.nodes[n][1])}' for n in G.nodes]
    node_topic = []
    node_name = []
    node_count = []
    for n in G.nodes:
        node_topic.append(int(G.nodes[n][0]))
        node_name.append(topic_dict[int(G.nodes[n][0])])
        node_count.append(int(G.nodes[n][1]))
    
    pos = nx.spring_layout(G, seed=1234)
    node_x = []
    node_y = []
    
    for n in pos:
        node_x.append(pos[n][0])
        node_y.append(pos[n][1])
        
    H = nx.DiGraph(G)
    H.remove_edges_from(nx.selfloop_edges(H))
    
    edge_x = []
    edge_y = []
    for edge in H.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
    
    edgewidth = [len(G.get_edge_data(u, v)) for u, v in G.edges()]
    nodesize = [G.nodes[n][1] for n in G.nodes]
    nodesize[-1] = 1
    edgewidth[-1] = 1
    fakesize = nodesize
    
    nodesize = [s*50 for s in normalized(nodesize)]
    # edgewidth = [w for w in normalized(edgewidth)]
    
    #create a trace for the edges
    trace_edges = go.Scatter(
        x=edge_x,
        y=edge_y,
        opacity=0.7,
        mode="lines+markers",
        marker=dict(color=edgewidth, 
                    size=10,
                    opacity=0.7,
                    symbol= "arrow-bar-up", 
                    angleref="previous",
                    colorscale='Plotly3'),
        hoverinfo='none',
        )

    #create a trace for the nodes
    trace_nodes = go.Scatter(
        x=node_x,
        y=node_y,
        text=node_topic,
        mode='markers',
        customdata=np.vstack((np.array(node_name), np.array(node_count))).T,
        marker=dict(symbol='circle',
                size=nodesize,
                color=fakesize,
                colorscale='Plotly3',
                colorbar=dict(
                    thickness=10,
                    title='Node Counts in session',
                    xanchor='left',
                    titleside='right'
                    )
                ),
        hovertemplate=
        "<b>Topic: %{text} </b><br>" +
        "<b>Name:</b> %{customdata[0]} <br>" +
        "<b>Count:</b> %{customdata[1]} <br>",
        )

    data = [trace_edges, trace_nodes]
    fig = go.Figure(data=data)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), showlegend=False)
    return fig