# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 12:58:37 2023

@author: Vivek
"""


import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import pickle
from sklearn import  metrics
import numpy as np
import plotly.graph_objs as go

#Define CSS style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


#Load data
df = pd.read_csv('Vivek.csv')
df['Date'] = pd.to_datetime (df['Date']) # create a new column 'data time' of datetime type
df2=df.iloc[:,2:7]
X2=df2.values
fig1 = px.line(df, x="Date", y=df.columns[1:7])# Creates a figure with the raw data


trace1 = go.Box(x=df['Power_kW'], name='Power')
trace2 = go.Box(x=df['Hour'], name='Hour')
trace3 = go.Box(x=df['temp_C'], name='Temperature C')
trace4 = go.Box(x=df['solarRad_W/m2'], name='Solar Radiation')

# Create a dictionary of options for the dropdown
options = {'Power': trace1, 'Hour': trace2, 'Temperature': trace3,'Solar Radiation': trace4}

df_real = pd.read_csv('real.csv')
df_real['Date'] = pd.to_datetime (df_real['Date'])
y2=df_real['Power_kW'].values


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X2 = sc.fit_transform(X2)


#Load and run LR model

with open('reg_L.pkl','rb') as file:
    LR_model2=pickle.load(file)

y2_pred_LR = LR_model2.predict(X2)



#Evaluate errors
MAE_LR=metrics.mean_absolute_error(y2,y2_pred_LR) 
MBE_LR=np.mean(y2-y2_pred_LR)
MSE_LR=metrics.mean_squared_error(y2,y2_pred_LR)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y2,y2_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y2)
NMBE_LR=MBE_LR/np.mean(y2)
#Load RF model
with open('reg_DT.pkl','rb') as file:
    RF_model2=pickle.load(file)

y2_pred_RF = RF_model2.predict(X2)

#Evaluate errors
MAE_RF=metrics.mean_absolute_error(y2,y2_pred_RF)
MBE_RF=np.mean(y2-y2_pred_RF) 
MSE_RF=metrics.mean_squared_error(y2,y2_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y2,y2_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y2)
NMBE_RF=MBE_RF/np.mean(y2)

# Create data frames with predictin results and error metrics 
d = {'Methods': ['Linear Regression','Decision Tree'], 'MAE': [MAE_LR, MAE_RF],'MBE': [MBE_LR, MBE_RF], 'MSE': [MSE_LR, MSE_RF], 'RMSE': [RMSE_LR, RMSE_RF],'cvMSE': [cvRMSE_LR, cvRMSE_RF],'NMBE': [NMBE_LR, NMBE_RF]}
df_metrics = pd.DataFrame(data=d)
d={'Date':df['Date'].values, 'LinearRegression': y2_pred_LR,'Decision Tree': y2_pred_RF}
df_forecast=pd.DataFrame(data=d)

# merge real and forecast results and creates a figure with it
df_results=pd.merge(df_real,df_forecast,on='Date')

fig2 = px.line(df_results,x=df_results.columns[0],y=df_results.columns[1:4])

fig6 = px.scatter(df_results,x=df_results.columns[1],y=df_results.columns[2])
fig7 = px.scatter(df_results,x=df_results.columns[1],y=df_results.columns[3])

# Define auxiliary functions
def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.layout = html.Div([
    html.H1('IST South Building Energy Forecast Tool (kWh)',style={'backgroundColor': '#000000','color': 'white','textAlign': 'center'}),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='RAW DATA', value='tab-1',style={
                'backgroundColor': '#0000FF',
                'color': 'black',
                'fontWeight': 'bold'}),
        dcc.Tab(label='DATA EXPLORATION', value='tab-4',style={
                'backgroundColor': '#00FF00',
                'color': 'black',
                'fontWeight': 'bold'
            }),
        dcc.Tab(label='FORECAST', value='tab-2',style={
                'backgroundColor': '#0000FF',
                'color': 'black',
                'fontWeight': 'bold'}),
        dcc.Tab(label='ERROR METRICS', value='tab-3',style={
                'backgroundColor': '#00FF00',
                'color': 'black',
                'fontWeight': 'bold'
            }),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H4('South Tower Raw Data 2019',style={
        'color': 'blue'}),
            dcc.Graph(
                id='yearly-data',
                figure=fig1,
            ),
        ])
    elif tab == 'tab-4':
        return html.Div([
html.H4('Please Select the Variable for BoxPlot',style={
'color': 'green'}),dcc.Dropdown(
        id='my-dropdown',
        options=[{'label': k, 'value': k} for k in options.keys()],
        value='Power'
    ),
    dcc.Graph(id='my-graph')
])
    
    elif tab == 'tab-2':
        return html.Div([
            html.H4('South Tower Electricity Forecast (kWh)',style={
        'color': 'blue'}),
            dcc.Graph(
                id='yearly-data',
                figure=fig2,
                ),
            
        ])
    elif tab == 'tab-3':
        return html.Div([html.H4('Please Select the type of Result Output you want from Dropdown List',style={
        'color': 'green'}),
    dcc.Dropdown(
        id='dropdown',
        options=[{'label': i, 'value': i} for i in ['Scatter Plot - Linear Regression', 'Scatter Plot - Decision Tree','Error Metrics Table']],
        value='Graph',
    ),
    html.Div(id='output-container')
])
    
    
@app.callback(
    dash.dependencies.Output('my-graph', 'figure'),
    [dash.dependencies.Input('my-dropdown', 'value')])
def update_graph(selected_option):
    return {'data': [options[selected_option]]}

@app.callback(
    dash.dependencies.Output('output-container', 'children'),
    [dash.dependencies.Input('dropdown', 'value')])
def update_output(value):
    if value == 'Scatter Plot - Linear Regression':
        return dcc.Graph(figure=fig6)
    elif value == 'Scatter Plot - Decision Tree':
        return dcc.Graph(figure=fig7)
    elif value == 'Error Metrics Table':
        return generate_table(df_metrics)


if __name__ == '__main__':
    app.run_server()
