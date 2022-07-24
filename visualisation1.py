# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 09:55:11 2022

@author: 2002
"""
import dash
from dash import dcc
from dash import html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd
import dash_bootstrap_components as dbc
import plotly.express as px

df2 = pd.read_excel("Dashboard_Database.xlsx",sheet_name=None)
df2a = df2.get('Segment_Wise').dropna(axis=1, how='all')
df2a = df2a.melt(id_vars='Date')
df2a['Date'] = pd.to_datetime(df2a.Date,format='%Y%m')+MonthEnd(0)
df2a['Exchange'] = df2a['variable'].str.split("_").str[0]
df2a['Segment'] = df2a['variable'].str.split("_").str[-1]
df2a['Segment'] = np.where(df2a['Segment'].str.contains("precious", case=False),"Others",df2a['Segment'])
df2a['Segment1'] = np.where(df2a['Segment'].str.contains("Agriculture|Bullion|Metals|Energy", case=False),df2a['Segment'],"Others")
df2a['Segment1'] = np.where(df2a['Segment'].str.contains("Agriculture|Bullion|Metals|Energy", case=False),df2a['Segment'],"Others")
s = df2a[['Segment1','Date','Exchange','value']].groupby(['Segment1','Date'])[['value']].sum().assign(Exchange = 'All Exchanges').set_index('Exchange',append=True)
df2b = pd.concat([df2a[['Segment1','Date','Exchange','value']],s.reset_index()]).sort_values(['Segment1','Date']).reset_index(drop=True)
s = df2b[['Segment1','Date','Exchange','value']].groupby(['Exchange','Date'])[['value']].sum().assign(Segment1 = 'All Segments').set_index('Segment1',append=True)
df2b = pd.concat([df2b[['Segment1','Date','Exchange','value']],s.reset_index()]).sort_values(['Exchange','Date']).reset_index(drop=True)
mgr_options1 = df2b["Exchange"].unique()
mgr_options2 = df2b["Segment1"].unique()
last_date = df2b.loc[len(df2b)-1,'Date'].strftime("%B %Y")

df3 = df2.get('Exchange_Wise').dropna(axis=1, how='all')
df3a = df3.melt(id_vars='Date')
df3a['Date'] = pd.to_datetime(df3a.Date,format='%Y%m')+MonthEnd(0)
df3a['Segment2'] = df3a['variable'].str.split("_").str[0]
df3a['FO'] = df3a['variable'].str.split("_").str[1]
df3a['Exchange'] = df3a['variable'].str.split("_").str[2]
df3b = df3a.groupby(['Date','Segment2'])[['value']].sum().unstack().apply(lambda x: round(x / x.sum() * 100, 2), axis=1)
df3b.columns = ['Agri','NonAgri']
df3b = df3b.tail(12).reset_index().melt(id_vars="Date")

df3c = df3a.groupby(['Date','FO'])[['value']].sum().unstack().apply(lambda x: round(x / x.sum() * 100, 2), axis=1)
df3c.columns = ['Fut','Opt']
df3c = df3c.tail(12).reset_index().melt(id_vars="Date")
def fig1():
    fig= px.line(df3b, x='Date', y='value', color='variable', markers=True)
    fig.update_layout(width=400)
    return fig

def fig2():
    fig= px.line(df3c, x='Date', y='value', color='variable', markers=True)
    fig.update_layout(width=400)
    return fig


app = dash.Dash()


app.layout = html.Div([
   html.Div(
        children=[
            html.H1('Dashboard for Commodity Derivatives Market for {}'.format(last_date), 
                    style={'textAlign': 'center','color': 'red', 'fontSize': 30})
      ]), 
    
   
    html.H2("Monthly Turnover Report", style={'color': 'royalblue'}),
    
    html.Div([ dcc.Markdown(id="markdown-result")
    ], style={'textAlign': 'left','color': 'black', 'fontSize': 20}),
    
    html.Div(
        # add some markdown text
    
        [dcc.Dropdown(
                id="my_exchange",
                options=[{
                    'label': i,
                    'value': i
                } for i in mgr_options1],
#                multi=True, placeholder='Select id...',
                value='All Exchanges'),
         ], style={'width': '48%', 'display': 'inline-block'}),
      
    html.Div(
          [dcc.Dropdown(
                    id="my_segment",
                    options=[{
                        'label': i,
                        'value': i
                    } for i in mgr_options2],
#                multi=True, placeholder='Select id...',
                value='All Segments'),
                  
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),  
    html.Br(),
    dcc.Graph(id='funnel-graph'),
 
    html.Br(),
    
    
    dbc.Container(fluid=True,children=[
        dbc.Row(
            [
                dbc.Col(children=[
                                html.H3('% Share of Agri and Non-Agri Segment'),
                                html.Div(children=dcc.Graph(figure= fig1() ))],style={'width':'50%', 'display': 'inline-block','background-color': 'lightblue'}),
                dbc.Col(children=[
                                html.H3('% Share of Futures and Options'),
                                html.Div(children=dcc.Graph(figure= fig2() ))],style={'width':'50%', 'display': 'inline-block','background-color': 'orange'}),
#                dbc.Col(html.Div("One of three columns"),style={'background-color': 'lightgreen'}, width=4),
            ],
        ),
    ])

    
])


def filter_data(Exchange, Segment):
    if (Exchange == "All Exchanges"):
        df2c= df2b[df2b['Exchange'] == "All Exchanges"].reset_index(drop=True)
        if Segment == "All Segments":
            return df2c
        else:
            return df2c[df2c['Segment1'] == Segment].reset_index(drop=True)  
    else:
        df2c= df2b[df2b['Exchange'] == Exchange].reset_index(drop=True)
        if Segment == "All Segments":
            return df2c
        else:
            return df2c[df2c['Segment1'] == Segment].reset_index(drop=True)

@app.callback(
    [Output("markdown-result", "children"),
    Output(component_id='funnel-graph', component_property='figure')],
    [Input(component_id='my_exchange', component_property='value'),
    Input(component_id='my_segment', component_property='value')]
    )
def update_graph(sel_exchange, sel_segment):
    df_plot = filter_data(sel_exchange, sel_segment)
    pv = pd.pivot_table(
    df_plot,
    index=['Date'],
    columns=["Segment1"],
    values=['value'],
    aggfunc=sum,
    fill_value=0)
        
    if "Agriculture" in list(pv.columns.levels[1]):
        trace1 = go.Bar(x=pv.index, y=pv[('value', 'Agriculture')], name='Agriculture')
    else:
        trace1 = {}
    
    if "Bullion" in list(pv.columns.levels[1]):
        trace2 = go.Bar(x=pv.index, y=pv[('value', 'Bullion')], name='Bullion')
    else:
        trace2 = {}
        
    if "Metals" in list(pv.columns.levels[1]):
        trace3 = go.Bar(x=pv.index, y=pv[('value', 'Metals')], name='Metals')
    else:
        trace3 = {}
            
    if "Energy" in list(pv.columns.levels[1]):
        trace4 = go.Bar(x=pv.index, y=pv[('value', 'Energy')], name='Energy')
    else:
        trace4 = {}
        
    if "Others" in list(pv.columns.levels[1]):
        trace5 = go.Bar(x=pv.index, y=pv[('value', 'Others')], name='Others')
    else:
        trace5 = {}
    
    df2a = filter_data("All Exchanges", 'All Segments').groupby(['Date'])[['value']].sum()
    change_turn = round((df2a.iloc[-1]/df2a.iloc[-2]-1) * 100,1)
    change_turn1 = str(np.where(change_turn.value>0, "increased by ",'Decreased by'))+str(change_turn.value)
    change_turn_yoy = round((df2a.iloc[-1]/df2a.iloc[-13]-1) * 100,1)
    change_turn_yoy1 = str(np.where(change_turn_yoy.value>0, "increased by ",'Decreased by'))+str(change_turn_yoy.value)

    output1 = 'The monthly turnover in commodity derivatives market in India {}% during {}. Year on year the turnover {}%'.format(change_turn1, last_date, change_turn_yoy1) 
    output2 = {
       'data': [trace4, trace2, trace3, trace1, trace5],
       'layout': go.Layout(
           title='Commodity Derivatives Turnover Report For {}'.format(sel_exchange),
           barmode='stack')}
    return [output1, output2]


      
if __name__ == '__main__':
    app.run_server()
