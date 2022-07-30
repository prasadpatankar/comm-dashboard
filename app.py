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
last_date_a = df2b.loc[len(df2b)-1,'Date']
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

df36 = pd.read_csv(r'mcx_close_prices.csv')
df36['Date'] = pd.to_datetime(df36['Date'])
df36['Symbol'] = df36['Symbol'].str.upper().str.strip()
#last_date1 = df36['Date'].iloc[-1]
df36 = df36[ df36['Date'] == last_date_a].reset_index(drop=True)
l1 = ["GOLD", "SILVER"]
df36['Symbol1'] = np.where(df36['Symbol'].str.contains('gold|silver', case=False),np.where(df36['Symbol'].isin(l1), df36['Symbol'] , np.nan),df36['Symbol']   )
df36 = df36.dropna().sort_values(by='mom', ascending=False).reset_index(drop=True)
df36['Segment'] = np.where(df36['Symbol'].str.contains('gold|silver', case=False),"Bullion",
                          np.where(df36['Symbol'].str.contains('kapas|cotton|mentha|rubber', case=False),"Agri",
                                   np.where(df36['Symbol'].str.contains('gas|crude', case=False),"Energy", "Metals" )))                              
df37 = pd.read_csv(r'ncdex_close_prices.csv')
df37 = df37.dropna()
df37.iloc[:,2] = df37.iloc[:,2].str.replace(",","").apply(float)
df37.iloc[:,1] = df37.iloc[:,1].str.replace(",","").apply(float)
df37['pct_change'] = df37.iloc[:,2]/df37.iloc[:,1]-1
df37.sort_values(by='pct_change', inplace=True)

df38 = pd.read_csv(r'ncdex_turnover.csv')
df38.iloc[:,1] = df38.iloc[:,1].str.replace(",","").apply(float)
df38 = df38.groupby(['Symbol'])[['Turnover']].sum().reset_index().sort_values(by="Turnover", ascending=False)          
df38['Turnover'] = (df38['Turnover'] / 
                      df38['Turnover'].sum()) * 100
df38a = df38[:5].copy()
new_row = pd.DataFrame(data = {
    'Symbol' : ['Others'],
    'Turnover' : [df38['Turnover'][5:].sum()]
})
df38b = pd.concat([df38a, new_row])

df39 = pd.read_csv(r'mcx_turnover.csv')
df39.iloc[:,1] = df39.iloc[:,1].str.replace(",","").apply(float)
df39['Symbol'] = np.where(df39['Symbol'].str.contains('gold', case=False),"Gold",
         np.where(df39['Symbol'].str.contains('silver', case=False),"Silver",df39['Symbol']))                       
df39 = df39.groupby(['Symbol'])[['Turnover']].sum().reset_index().sort_values(by="Turnover", ascending=False)          
df39['Turnover'] = (df39['Turnover'] / 
                      df39['Turnover'].sum()) * 100
df39a = df39[:5].copy()
new_row = pd.DataFrame(data = {
    'Symbol' : ['Others'],
    'Turnover' : [df39['Turnover'][5:].sum()]
})
df39b = pd.concat([df39a, new_row])

def fig1():
    text1 = [f'{round(df3b.iloc[i,2],1)}%' if df3b.iloc[i,0]== last_date_a else "" for i in df3b.index]
    fig= px.line(df3b, x='Date', y='value', color='variable', text=text1, markers=True, template='simple_white', width=700)
    fig.update_yaxes(title_text="Percent Change")
    fig.update_xaxes(title_text="")
    fig.update_traces(textposition='top left')
    return fig

def fig2():
    text1 = [f'{round(df3c.iloc[i,2],1)}%' if df3c.iloc[i,0]== last_date_a else "" for i in df3c.index]
    fig= px.line(df3c, x='Date', y='value', color='variable', text=text1, markers=True, template='simple_white', width=700)
    fig.update_yaxes(title_text="Percent Change")
    fig.update_xaxes(title_text="")
    fig.update_traces(textposition='top left')
    return fig

def fig3():
    text1 = [f'{round(i*100,1)}%' for i in df36['mom']]
    fig3 = px.bar(df36, x='mom',  y='Symbol', color='Segment', text=text1, template='simple_white', width=700)#title='<b>Monthly Percent Change in the Commodity Prices at MCX</b>', 
    #fig2.update_traces(marker_color='#3EB595')
    fig3.update_layout( uniformtext_minsize=10,  autosize=True, title_x=0.3, title_font_family ="Calibri")
    fig3.update_layout(xaxis= { 'tickformat': ',.0%'}, hovermode='closest', legend_orientation="v", legend=dict(
            x= 1,
            y=0.9,))
    fig3.update_traces(textposition="outside")
    fig3.update_xaxes(title_text="Percent Change")
    fig3.update_yaxes(title_text="")
    return fig3

def fig4():
    text1 = [f'{round(i*100,1)}%' for i in df37['pct_change']]
    fig3 = px.bar(df37, x='pct_change',  y='Symbol',  text=text1,  template='simple_white', width=700)#, title='<b>Monthly Percent Change in the Commodity Prices at NCDEX</b>',
    #fig2.update_traces(marker_color='#3EB595')
    fig3.update_layout( uniformtext_minsize=10,  autosize=True, title_x=0.3, title_font_family ="Calibri")
    fig3.update_layout(xaxis= { 'tickformat': ',.0%'}, hovermode='closest', legend_orientation="v", legend=dict(
            x= 1,
            y=0.9,))
    fig3.update_traces(textposition="outside")
    fig3.update_xaxes(title_text="Percent Change")
    fig3.update_yaxes(title_text="")
    return fig3

def fig56(data):
    labels = data['Symbol']
    values = data['Turnover']
    colors = ['blue','red','lightblue','orange','green']
    fig6 = go.Figure(data = go.Pie(values = values, 
                                   labels = labels, hole = 0.7, direction ='clockwise', sort=True,
                                   marker_colors = colors ))
    fig6.update_traces(textposition = 'outside', hoverinfo='label+percent',
                       textinfo='percent+label', textfont_size=16)
    #fig.update_traces(textposition = 'outside' , textinfo = 'percent+label')
    fig6.update_layout(showlegend=False,
 #                      title_text = '% Share of Top 5 Commodities in NCDEX Turnover',
                       title_font = dict(size=20,family='Verdana', 
                                         color='darkred'))
    fig6.add_annotation(x= 0.5, y = 0.5,
                        text = last_date,
                        font = dict(size=17,family='Verdana', 
                                    color='black'),
                        showarrow = False)
    return fig6




app = dash.Dash()
server = app.server


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
        dbc.Row(
            [
                dbc.Col(children=[
                                html.H3('Commodity Prices at MCX')
                    ,
                                html.Div(children=dcc.Graph(figure= fig3() ))],style={'width':'50%', 'display': 'inline-block','background-color': 'lightblue'}),
                dbc.Col(children=[
                                html.H3('Commodity Prices at NCDEX'),
                                html.Div(children=dcc.Graph(figure= fig4() ))],style={'width':'50%', 'display': 'inline-block','background-color': 'orange'}),
#                dbc.Col(html.Div("One of three columns"),style={'background-color': 'lightgreen'}, width=4),
            ],
        ),

        dbc.Row(
            [
                dbc.Col(children=[
                                html.H3('% Share of Top 5 Commodities at MCX')
                    ,
                                html.Div(children=dcc.Graph(figure= fig56(df39b) ))],style={'width':'50%', 'display': 'inline-block','background-color': 'lightblue'}),
                dbc.Col(children=[
                                html.H3('% Share of Top 5 Commodities at NCDEX'),
                                html.Div(children=dcc.Graph(figure= fig56(df38b) ))],style={'width':'50%', 'display': 'inline-block','background-color': 'orange'}),
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
