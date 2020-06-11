import pandas as pd
import numpy as np

dfb = pd.read_csv('https://raw.githubusercontent.com/bchap90210/bchap90210/Data-Files/acs2017_census_tract_data.csv')

cenData = dfb

stateP = cenData[['State','TotalPop','Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific','Poverty', 'ChildPoverty', 'Professional', 'Service', 'Office','Construction','Production','Drive','Carpool','Transit','Walk','OtherTransp','WorkAtHome', 'PrivateWork', 'PublicWork', 'SelfEmployed', 'FamilyWork', 'Unemployment']]

percentages = ['Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific','Poverty', 'ChildPoverty', 'Professional', 'Service', 'Office','Construction','Production','Drive','Carpool','Transit','Walk','OtherTransp','WorkAtHome', 'PrivateWork', 'PublicWork', 'SelfEmployed', 'FamilyWork', 'Unemployment']
for i in percentages:
    stateP[i] = round(stateP['TotalPop'] * stateP[i] / 100) 

stateDF = stateP.groupby(['State']).sum()

import numpy as np
clusters = pd.read_csv("https://raw.githubusercontent.com/amanso13/covid-dash-manso/master/Clusters.csv")
y_kmeans = clusters.Cluster.values
stateDF['cluster'] = y_kmeans

##############################################################################
#Bringing in Covid Cases by State#############################################
##############################################################################
# Reading in the data
df2 = pd.read_csv('https://raw.githubusercontent.com/amanso13/covid-dash-manso/master/CoreData.csv', header=0, encoding='latin-1')

stsumDF = df2

# Joining the two DFs
joinDF = stateDF.merge(stsumDF, on='State', how='left',)

joinDF.dropna(inplace=True)

# Sorting the data
joinDF.sort_values(by=['Confirmed'])


###################### MANSO ########################

joinDF = joinDF.groupby(["State"]).max().reset_index()


########################################################################################################################################################



# Various Corn Dashboard

# In[Load Packages]

import pandas as pd
import plotly.express as px
from plotly.graph_objs import *
import plotly.graph_objs as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output
from datetime import datetime as dt
import numpy as np


# In[Data Load and Prep]

df = pd.read_csv("https://raw.githubusercontent.com/amanso13/covid-dash-manso/master/CoreData.csv")
df = df.drop("Unnamed: 0", axis = 1)
# Need to run Beau's code first
df = df.merge(joinDF[['State', 'cluster', 'TotalPop']], how = "left", on = "State")
df.columns = ['State Abb', 'State', 'Date', 'Lat', 'Lon', 'Confirmed', 'Deaths',
       'Recovered', 'Active', 'Incident Rate', 'People Tested',
       'People Hospitalized', 'Mortality Rate', 'Testing Rate',
       'Hospitalization Rate', 'cluster', 'Total Population']
df2 = df.groupby('State').max().reset_index()
df2.cluster[df2.State == "California"] = 0
df2.cluster = df2.cluster.astype('category')
# fig.write_html("Plotly/line1.html")
df1 = df.groupby(["State Abb"]).max().reset_index()

mystates = list(df1.State.values)
mystates.insert(0, "Select All")
mystatesabb = list(df1["State Abb"].values)
mystatesabb.insert(0, "Select All")


myfeatures = list(df1.columns[5:-2])

fig_clus = px.scatter_3d(df2, x = 'Confirmed', y = 'Total Population', z = 'Deaths', color='cluster', text = 'State Abb', opacity=0.8, color_discrete_sequence=['#D44500','#3E3D3C','#283189'])
# fid_clus.write_html("Plotly/clus1.html")

#fig_rect = go.Figure(go.Scatter(x=[3,5,5,3], y=[1,1,2,2], fill="toself",fillcolor='#3E3D3C'))
#fig_rect.update_xaxes(range=[3,5],showgrid=False, zeroline = False, visible = False)
#fig_rect.update_yaxes(range=[1,2],showgrid=False, zeroline = False, visible = False)
#fig.write_html("Plotly/rect1.html")

df_log = df[["State Abb","Date","Confirmed","Deaths"]]
df_log.Confirmed = np.log(df_log.Confirmed)
df_log.Deaths = np.log(df_log.Deaths)
df_log.Confirmed[df_log.Confirmed < 0] = 0
df_log.Deaths[df_log.Deaths < 0] = 0
df_log = df_log.sort_values(by="Date")
df_log = df_log[df_log['State Abb'] != "NY"]
#fig_scat = px.scatter(df_log[df_log.Date > "2020-03-01"], x = 'Confirmed', y = 'Deaths', text = 'State Abb', size = "Confirmed", color = "State Abb", animation_frame = "Date", animation_group = "State Abb",range_y=[0,12],range_x=[0,15])
# fig_scat.write_html("Plotly/scat1.html")

df_choro = df.sort_values(by="Date")
#tmp_choro = px.choropleth(df_choro, locations='State Abb', locationmode="USA-states", color='Confirmed', scope="usa", hover_data=["Confirmed","Deaths", "Recovered"], color_continuous_scale=['#283189','#ADB3B8','#D44500'], animation_frame = "Date", animation_group = "State Abb").update_layout(margin=dict(l=0, r=0, t=0, b=0, pad=0), paper_bgcolor='rgba(232,234,235,.3)', plot_bgcolor='rgba(232,234,235,0)',geo=dict(bgcolor= 'rgba(0,0,0,0)'))
# tmp_choro.write_html("Plotly/choro_ani.html")


dfm = pd.read_csv("https://raw.githubusercontent.com/amanso13/covid-dash-manso/master/Global_Mobility_Report.csv")
dfm = dfm.merge(df1[["State","State Abb"]],how="left",on="State")
dfm.Date = pd.to_datetime(dfm.Date)
dfm["State Abb"] = dfm["State Abb"].fillna("Avg")
# mob = px.line(dfm[dfm.State != "Average"], x='Date',y='Retail and Rec', color='State')
# mob.add_scatter(x=dfm['Date'][dfm.State == "Average"],y=dfm['Retail and Rec'][dfm.State == "Average"], mode='lines',name="US Average",line=dict(color="black",width=6, dash='dot'))
#mob.write_html("Plotly/mob1.html")
mobtypes = list(dfm.columns[2:-1])

#mob = px.line(x=dfm['Date'][dfm.State == "Average"],y=dfm['Retail and Rec'][dfm.State == "Average"],color_discrete_sequence=["black"]).update_traces(line=dict(width=6,dash='dot'))
# mob.write_html("Plotly/mob1.html")

tot_cases = sum(df1.Confirmed)
tot_deaths = sum(df1.Deaths)

pred = pd.read_csv("https://raw.githubusercontent.com/amanso13/covid-dash-manso/master/prophet_all.csv")
pred.Date = pd.to_datetime(pred.Date)
pred = pd.melt(pred, id_vars = ["Date"])
pred[['State Abb','variable']] = pred.variable.str.split('_',expand=True)
pred = pred.merge(df1[['State Abb',"State"]],how='left',on='State Abb')

#pred_chart = px.line(pred, x='Date',y="value", color_discrete_sequence=["firebrick"])
#pred_chart.add_scatter(x=pred.Date,y=pred.AL_lower, mode='lines',name='Lower', line=dict(color="grey", dash="dot"))
#pred_chart.add_scatter(x=pred.Date,y=pred.AL_upper, mode='lines',name='Upper',fill='tonexty',line=dict(color="grey", dash="dot"))
#pred_chart.add_scatter(x=df.Date[df.State=="Alabama"],y=df.Confirmed[df.State=="Alabama"], mode='lines', name='Upper', line=dict(color="steelblue")).update_layout(margin=dict(l=10, r=10, t=10, b=10, pad=0), paper_bgcolor='rgba(232,234,235,.5)', plot_bgcolor='rgba(232,234,235,0)',geo=dict(bgcolor= 'rgba(0,0,0,0)'),showlegend=False)
#pred_chart.write_html("Plotly/pred1.html")



# In[Dashboard]

################ START APP ################
#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)
server = app.server
app.layout = html.Div([
    html.H1("VARIOUS CORN", style={'text-align':'center', 'font-size':64, 'margin-block-end':0, 'margin-block-start':0,'color':'#3E3D3C'}),
    html.H2("COVID-19 | Dashboard", style={'text-align':'center', 'font-size':36, 'margin-block-end':5, 'margin-block-start':0,'color':'#D44500','border-bottom':'3px solid #6F777D'}),
    html.Img(src="https://styleguide.ischool.syr.edu/img/logo-thumb-block-syracuse-white-gray.png", style={'float':'right', 'height':'7.5%','width':'6.5%', 'position':'relative', 'top':'-125px','right':'10px'}),
    
    html.Div([
    html.H3("Select Values from the Dropdown Menus:", style={'font-family':"Sherman Serif", 'text-align':'left', 'margin-block-end':1, 'margin-block-start':20, 'marginLeft':'10px','display':'inline-block','position':'relative','top':"0px"}),   
    
    ],className="tableDiv",style={'marginBottom':'2px','height':'50px'}),
    html.Div([
    html.Div([
        html.Div([
        dcc.DatePickerRange(id='my-date-picker-range',min_date_allowed=dt(2020, 1, 22),
                            max_date_allowed=dt(2020, 6, 6), initial_visible_month=dt(2020,6, 1),
                            start_date=dt(2020, 1, 22), end_date=dt(2020, 6, 6)),
        html.Div(id='output-container-date-picker-range')], className="row",
        style={'marginTop': 0, 'marginBottom': 5, 'marginLeft': 5, 'font-family':"Sherman Serif",'display': 'inline-block','position':"relative",'top':"-21px"}),
        html.Div(dcc.Dropdown(id='dropdown1', options=[{'label': i, 'value': i} for i in myfeatures],
                           value="Confirmed", style={'width': '180px','height':"48px"}),style={'display': 'inline-block', 'marginLeft':5,'marginTop': 5}),
        html.Div(dcc.Dropdown(id='mob_type', options=[{'label': i, 'value': i} for i in mobtypes],
                           value="Retail and Rec", style={'width': '180px','height':"48px"}),style={'display': 'inline-block', 'marginLeft':5,'marginTop': 5}),
            html.Div(dcc.Dropdown(id='dropdown', multi=True, options=[{'label': i, 'value': i} for i in mystates], value="Select All", style={'width': '545px',"height":'48px'}),style={'display': 'inline-block','marginLeft':5,'marginTop': 5}),
            
            html.Div(html.Div(id="total",style={'font-family':"Sherman Serif", 'text-align':'left', "font-size": "22px", 'font-weight': 'bold', 'marginLeft':'10px','marginTop':'2px','vertical-align': 'middle'}),style={'background-color':'rgba(232,234,235,.5)', 'height':'58px', 'width': '597px', 'marginTop': 1,'border':'1px solid #ADB3B8','position':"relative","top":'-80px',"right":'-1225px'})
        
        ],style={'background-color':'rgba(232,234,235,.4)','height':'58px', 'width': '1215px','marginLeft': 5, 'marginTop': 1,'marginBottom': 5,'border':'1px solid #ADB3B8'}),
        ]),
    html.Div([dcc.Graph('fig', config={'displayModeBar': False}, style={'width': '600px', 'height': '300px', 'marginLeft':10,'marginBottom':10,'display': 'inline-block','vertical-align': 'top'}),dcc.Graph('MODEL_PLACEHOLDER', config={'displayModeBar': False}, style={'width': '600px', 'height': '600px', 'marginLeft':10,'marginBottom':10,'marginTop':1,'display': 'inline-block'}),dcc.Graph('fig1', config={'displayModeBar': False}, style={'width': '600px', 'height': '300px', 'marginLeft':10,'display': 'inline-block','vertical-align': 'top'}),dcc.Graph('choro_ani', config={'displayModeBar': False}, style={'width': '600px', 'height': '295px', 'marginLeft':10,'marginBottom':10,'marginTop':1,'position':'relative','top':'-315px','display': 'inline-block'}),dcc.Graph('mob',config={'displayModeBar': False}, style={'width': '600px', 'height': '295px', 'marginLeft':10,'display': 'inline-block','position':'relative',"top":'-325px','right':'-610px'}),],className="row",style={'width': '100%', 'display': 'inline-block','height': '610px'}),
    html.Div([
        html.Div(html.Iframe(src=f"https://flo.uri.sh/visualisation/2498856/embed", style={'width': '595px', 'height': '595px'}), style={'display':'inline-block', 'marginLeft':10}, className="six columns"),
        html.Div(html.Iframe(src=f"https://flo.uri.sh/visualisation/2770197/embed", style={'width': '597px', 'height': '595px'}), style={'display':'inline-block', 'marginLeft':10, 'marginBottom':10,'marginTop':10}, className="six columns"),
        html.Div(dcc.Graph('figClus', config={'displayModeBar': False}, style={'width': '600px', 'height': '598px'}), style={'display':'inline-block', 'marginLeft':10, 'marginTop':10}, className="six columns"),
        ], className="row",style={'height': '610px','position':'relative','top':'-330px'}),
],style={'font-family':"Sherman Serif",'background-image':'url(https://www.lsi.umich.edu/sites/default/files/styles/paragraph_float_image/public/media/shared/AdobeStock_330293007.jpeg?itok=HkRYAj8g)','background-size': 'cover','color':'#3E3D3C','height':'1500px'})
#,'background-size': 'cover', 'left':'-1000px'

@app.callback(Output('datatable', 'data'),
 	[Input('my-date-picker-range', 'start_date'),
	 Input('my-date-picker-range', 'end_date')]
)

@app.callback(
    Output('fig', 'figure'),
    [Input('dropdown', 'value'),
     Input('dropdown1', 'value'),
     Input('my-date-picker-range', 'end_date')]
)


def update_graph(state_name, myfeat, end_date):
    import plotly.express as px
    end_date = end_date[:10]
    if (state_name == "Select All") | (state_name == ["Select All"]):
        return px.choropleth(data_frame = df[df.Date == end_date], locations='State Abb', locationmode="USA-states", color=myfeat, scope="usa", hover_data=["Confirmed","Deaths", "Recovered"], color_continuous_scale=['#283189','#ADB3B8','#D44500']).update_layout(margin=dict(l=0, r=0, t=0, b=0, pad=0), paper_bgcolor='rgba(232,234,235,.3)', plot_bgcolor='rgba(232,234,235,0)',geo=dict(bgcolor= 'rgba(0,0,0,0)'))
    else:
        return px.choropleth(data_frame = df[(df.State.isin(state_name)) & (df.Date == end_date)], locations="State Abb", locationmode="USA-states", color=myfeat, scope="usa", hover_data=["Confirmed","Deaths", "Recovered"],color_continuous_scale=['#283189','#ADB3B8','#D44500']).update_layout(margin=dict(l=0, r=0, t=0, b=0, pad=0), paper_bgcolor='rgba(232,234,235,.3)', plot_bgcolor='rgba(232,234,235,0)',geo=dict(bgcolor= 'rgba(0,0,0,0)')) #,coloraxis_showscale=False

@app.callback(
    Output('fig1', 'figure'),
    [Input('dropdown', 'value'),
     Input('my-date-picker-range', 'start_date'),
	 Input('my-date-picker-range', 'end_date'),
     Input('dropdown1', 'value')]
)
   
def update_graph1(stateabb_name, start_date, end_date, myfeat):
    import plotly.express as px
    if (stateabb_name == "Select All") | (stateabb_name == ["Select All"]):
        return px.scatter(data_frame = df[(df.Date >= start_date) & ( df.Date <= end_date)], x="Date", y=myfeat, color = "State Abb", size = myfeat).update_layout(margin=dict(l=10, r=10, t=10, b=10, pad=0), paper_bgcolor='rgba(232,234,235,.7)', plot_bgcolor='rgba(232,234,235,0)',geo=dict(bgcolor= 'rgba(0,0,0,0)'))
    else:
        return px.scatter(data_frame = df[(df.State.isin(stateabb_name)) & (df.Date >= start_date) & ( df.Date <= end_date)], x="Date", y=myfeat, color = "State Abb", size = myfeat).update_layout(margin=dict(l=10, r=10, t=10, b=10, pad=0), paper_bgcolor='rgba(232,234,235,.7)', plot_bgcolor='rgba(232,234,235,0)',geo=dict(bgcolor= 'rgba(0,0,0,0)'))


@app.callback(
    Output('figClus', 'figure'),
    [Input('dropdown', 'value')]
)

def update_graph2(stateabb_name):
    import plotly.express as px
    if (stateabb_name == "Select All") | (stateabb_name == ["Select All"]):
        return px.scatter_3d(df2, x = 'Confirmed', y = 'Total Population', z = 'Deaths', color='cluster', text = 'State Abb', opacity=0.8, color_discrete_sequence = ['#D44500','#3E3D3C','#283189']).update_layout(margin=dict(l=10, r=10, t=10, b=10, pad=0), paper_bgcolor='rgba(232,234,235,.5)', plot_bgcolor='rgba(232,234,235,0)',geo=dict(bgcolor= 'rgba(0,0,0,0)'),showlegend=False)
    else:
        return px.scatter_3d(df2[df2.State.isin(stateabb_name)], x = 'Confirmed', y = 'Total Population', z = 'Deaths', color='cluster', text = 'State Abb', opacity=0.8, color_discrete_sequence = ['#D44500','#3E3D3C','#283189']).update_layout(margin=dict(l=10, r=10, t=10, b=10, pad=0), paper_bgcolor='rgba(232,234,235,.5)', plot_bgcolor='rgba(232,234,235,0)',geo=dict(bgcolor= 'rgba(0,0,0,0)'),showlegend=False)


@app.callback(
    Output('choro_ani', 'figure'),
    [Input('dropdown1', 'value')]
)

def update_graph3(myfeat):
    import plotly.express as px
    return px.choropleth(data_frame = df_choro, locations='State Abb', locationmode="USA-states", color=myfeat, scope="usa", color_continuous_scale=['#283189','#ADB3B8','#D44500'], animation_frame = "Date", animation_group = "State Abb").update_layout(margin=dict(l=0, r=0, t=0, b=0, pad=0), paper_bgcolor='rgba(232,234,235,.3)', plot_bgcolor='rgba(232,234,235,0)',geo=dict(bgcolor= 'rgba(0,0,0,0)'),width = 600, height = 300)


@app.callback(
    Output('mob', 'figure'),
    [Input('dropdown', 'value'),
     Input('my-date-picker-range', 'start_date'),
	 Input('my-date-picker-range', 'end_date'),
     Input('mob_type','value')]
)

def update_graph4(state_name, start_date, end_date, mobtype_value):
    import plotly.express as px
    if (state_name == "Select All") | (("Select All" in state_name) & (len(state_name) == 1)):
        state_name = mystates
    if dt.strptime(start_date[:10], "%Y-%m-%d") < min(dfm.Date):
        start_date = min(dfm.Date)
    elif dt.strptime(start_date[:10], "%Y-%m-%d") > max(dfm.Date):
        start_date = max(dfm.Date)
    if dt.strptime(end_date[:10], "%Y-%m-%d") > max(dfm.Date):
        end_date = max(dfm.Date)
    elif dt.strptime(end_date[:10], "%Y-%m-%d") < min(dfm.Date):
        end_date = min(dfm.Date)
        
    mob = px.line(dfm[(dfm.State != "Average") & (dfm.State.isin(state_name)) & (dfm.Date >= start_date) & ( dfm.Date <= end_date)], x='Date',y=mobtype_value, color='State Abb')
    mob.add_scatter(x=dfm['Date'][(dfm.State == "Average") & (dfm.Date >= start_date) & ( dfm.Date <= end_date)],y=dfm[mobtype_value][dfm.State == "Average"], mode='lines',name="Avg", line=dict(color="black",width=5, dash='dot'))
    return mob.update_layout(margin=dict(l=10, r=10, t=10, b=10, pad=0), paper_bgcolor='rgba(232,234,235,.7)', plot_bgcolor='rgba(232,234,235,0)',geo=dict(bgcolor= 'rgba(0,0,0,0)'))


@app.callback(Output('total', 'children'),
              [Input('dropdown', 'value'),
               Input('dropdown1', 'value'),
               Input('my-date-picker-range', 'end_date')])

def update_table(stateabb_name, myfeat, end_date):
    if (stateabb_name == "Select All") | (("Select All" in stateabb_name) & (len(stateabb_name) == 1)):
        df_tmp = df.groupby('Date').agg({'Confirmed':'sum','Deaths':'sum','Recovered':'sum','Active':'sum',
                                        'Incident Rate':'sum', 'People Tested':'sum', 'People Hospitalized':'sum',
                                        'Mortality Rate':'mean','Testing Rate':'mean',
                                        'Hospitalization Rate':'mean'}).reset_index()
        df_tmp = df_tmp[df_tmp.Date==end_date[:10]]
        return 'The {} for the US on {} was {:,}.'.format(myfeat,end_date[:10],int(df_tmp[myfeat].iloc[0]))
    elif (("Select All" in stateabb_name) & (len(stateabb_name) > 1)):
        stateabb_name.remove("Select All")
        df_tmp = df[(df.State == stateabb_name[0]) & (df.Date == end_date[:10])]
        return 'The {} for {} on {} was {:,}.'.format(myfeat,df_tmp.State.iloc[0],end_date[:10],int(df_tmp[myfeat].iloc[0]))
    else: 
        df_tmp = df[(df.State == stateabb_name[0]) & (df.Date == end_date[:10])]
        return 'The {} for {} on {} was {:,}.'.format(myfeat,df_tmp.State.iloc[0],end_date[:10],int(df_tmp[myfeat].iloc[0]))


@app.callback(
    Output('MODEL_PLACEHOLDER', 'figure'),
    [Input('dropdown', 'value')]
)

def update_graph5(state_name):
    import plotly.express as px
    if (state_name == "Select All") | (("Select All" in state_name) & (len(state_name) == 1)):
        state_name = mystates
    predchart = px.line(pred, x=pred.Date[(pred.State.isin(state_name)) & (pred.variable == "Predict")], y=pred.value[(pred.State.isin(state_name)) & (pred.variable == "Predict")], color = pred['State Abb'][(pred.State.isin(state_name)) & (pred.variable == "Predict")])
    for i in range(len(state_name)):
        predchart.add_scatter(x=pred.Date[(pred.State==state_name[i]) & (pred.variable == "Predict")], y=pred.value[(pred.State==state_name[i]) & (pred.variable == "lower")], mode="lines", line=dict(color='rgba(111,119,125,.4)', dash='dot',width=4))
        predchart.add_scatter(x=pred.Date[(pred.State==state_name[i])& (pred.variable == "Predict")], y=pred.value[(pred.State==state_name[i]) & (pred.variable == "upper")], mode="lines",line=dict(color='rgba(111,119,125,.4)', dash='dot',width=4),fill='tonexty')
        predchart.add_scatter(x=df.Date[df.State==state_name[i]], y=df.Confirmed[df.State==state_name[i]], mode='lines', line=dict(color='rgba(111,119,125,.7)', width=3))
    return predchart.update_layout(margin=dict(l=10, r=10, t=10, b=10, pad=0), paper_bgcolor='rgba(232,234,235,.3)', plot_bgcolor='rgba(232,234,235,0)',geo=dict(bgcolor= 'rgba(0,0,0,0)'),showlegend=False,xaxis_title=None,yaxis_title=None)




if __name__ == '__main__':
    app.run_server(debug=True)

