from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

from lmao.visualizer.alice_info import generate_visualizations as generate_visualizations1
from lmao.visualizer.alice_info import get_constants as get_constants1
from lmao.visualizer.alice_info import get_graph as get_graph1

from lmao.visualizer.hdfs import generate_visualizations as generate_visualizations2
from lmao.visualizer.hdfs import get_constants as get_constants2
from lmao.visualizer.hdfs import get_graph as get_graph2

get_constants = get_constants1

# Define function to load data based on tab selection
def load_data(tab):
    if tab == 'alice':
        return get_constants1
    elif tab == 'hdfs':
        return get_constants2

total_log, total_sessions, total_topics, normal_anomaly_ratio = get_constants()

# Initialize the app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title='LMAO Data Visualizer Dashboard')
server = app.server

def generate_stats_card (title, value, image_path):
    return html.Div(
        dbc.Card([
            dbc.CardImg(src=image_path, top=True, style={'width': '50px','alignSelf': 'center'}),
            dbc.CardBody([
                html.P(value, className="card-value", style={'margin': '0px','fontSize': '22px','fontWeight': 'bold'}),
                html.H4(title, className="card-title", style={'margin': '0px','fontSize': '18px','fontWeight': 'bold'})
            ], style={'textAlign': 'center'}),
        ], style={'paddingBlock':'10px',"backgroundColor":'#636efa','border':'none','borderRadius':'10px'})
    )


tab_style = {
    'idle':{
        'borderRadius': '10px',
        'padding': '0px',
        'marginInline': '5px',
        'display':'flex',
        'alignItems':'center',
        'justifyContent':'center',
        'fontWeight': 'bold',
        'backgroundColor': '#636efa',
        'border':'none'
    },
    'active':{
        'borderRadius': '10px',
        'padding': '0px',
        'marginInline': '5px',
        'display':'flex',
        'alignItems':'center',
        'justifyContent':'center',
        'fontWeight': 'bold',
        'border':'none',
        'textDecoration': 'underline',
        'backgroundColor': '#636efa'
    }
}

big_graph_style = {'width': '98%', 
               'display': 'inline-block',
               'border-radius': '15px',
               'box-shadow': '8px 8px 8px grey',
               'background-color': '#f9f9f9',
               'padding': '10px',
               'margin-bottom': '10px',
               'margin-left': '10px'}
small_graph_style = {'width': '49%', 
               'display': 'inline-block',
               'border-radius': '15px',
               'box-shadow': '8px 8px 8px grey',
               'background-color': '#f9f9f9',
               'padding': '10px',
               'margin-bottom': '10px',
               'margin-left': '10px'}

# Define the layout of the app
app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.Img(src="/pic/assets/imdb.png",width=150), width=2),
            dbc.Col(
                dcc.Tabs(id='graph-tabs', value='alice', children=[
                    dcc.Tab(label='ALICE Infologger', value='alice',style=tab_style['idle'],selected_style=tab_style['active']),
                    dcc.Tab(label='HDFS log', value='hdfs',style=tab_style['idle'],selected_style=tab_style['active']),
                    dcc.Tab(label='BGL (upcoming)', value='bgl',style=tab_style['idle'],selected_style=tab_style['active'])
                ], style={'marginTop': '15px', 'width':'600px','height':'50px'})
            ,width=6),
        ]),
        dbc.Row([
            dbc.Col(generate_stats_card("Rows",total_log,"/pic/assets/movie-icon.png"), width=3),
            dbc.Col(generate_stats_card("Sessions", total_sessions,"/pic/assets/language-icon.svg"), width=3),
            dbc.Col(generate_stats_card("Topics",total_topics,"/pic/assets/country-icon.png"), width=3),
            dbc.Col(generate_stats_card("Normal:Anomaly",normal_anomaly_ratio,"/pic/assets/vote-icon.png"), width=3),
        ],style={'marginBlock': '10px'}),
        dbc.Row([
            dcc.Loading([
                html.Div(id='tabs-content')
            ],type='default',color='#636efa')
        ])
    ], style={'padding': '0px'})
],style={'backgroundColor': 'white', 'minHeight': '100vh'})

@app.callback(
    Output('tabs-content', 'children'),
    Input('graph-tabs', 'value')
)
def update_tab(tab):
    
    if tab == 'alice':
        fig1, fig2, fig3, fig4 = generate_visualizations1()
        fig5 = get_graph1()
        return html.Div([
        html.Div([
            dcc.Graph(id='graph5', figure=fig5, responsive=True),
        ], style=big_graph_style),
        html.Div([
            dcc.Graph(id='graph1', figure=fig1, responsive=True),
        ], style=small_graph_style),
        html.Div([
            dcc.Graph(id='graph2', figure=fig2, responsive=True),
        ], style=small_graph_style),
        html.Div([
            dcc.Graph(id='graph3', figure=fig3, responsive=True),
        ], style=big_graph_style),
        html.Div([
            dcc.Graph(id='graph4', figure=fig4, responsive=True),
        ], style=big_graph_style)
    ])
        
    elif tab == 'hdfs':
        fig1, fig2, fig3 = generate_visualizations2()
        fig5 = get_graph2()
        return html.Div([
        html.Div([
            dcc.Graph(id='graph5', figure=fig5, responsive=True),
        ], style=big_graph_style),
        html.Div([
            dcc.Graph(id='graph1', figure=fig1, responsive=True),
        ], style=small_graph_style),
        html.Div([
            dcc.Graph(id='graph2', figure=fig2, responsive=True),
        ], style=small_graph_style),
        html.Div([
            dcc.Graph(id='graph3', figure=fig3, responsive=True),
        ], style=big_graph_style)
    ])


if __name__ == '__main__':
    app.run_server(debug=False)