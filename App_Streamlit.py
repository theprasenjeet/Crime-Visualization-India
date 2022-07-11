#importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from matplotlib import cm
import matplotlib
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)    #THIS LINE IS MOST IMPORTANT AS THIS WILL DISPLAY PLOT ON
#NOTEBOOK WHILE KERNEL IS RUNNING


from IPython.display import HTML,display
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
from plotly.offline import init_notebook_mode

init_notebook_mode(connected=True)    #THIS LINE IS MOST IMPORTANT AS THIS WILL DISPLAY PLOT ON
#NOTEBOOK WHILE KERNEL IS RUNNING


import warnings
warnings.filterwarnings("ignore")


st.title("HotSpot Visualization")

uploaded_file = st.file_uploader("Choose a file 1:")
if uploaded_file is not None:
  sc1 = pd.read_csv(uploaded_file)
  st.write(sc1)

uploaded_file = st.file_uploader("Choose a file 2:")
if uploaded_file is not None:
  sc13 =pd.read_csv(uploaded_file)
  st.write(sc13)

frames = [sc1 , sc13]
sc = pd.concat(frames)
sc['STATE/UT'] = sc['STATE/UT'].str.capitalize()
sc['DISTRICT'] = sc['DISTRICT'].str.capitalize()
sc13.columns

sc13 = sc13[['STATE/UT', 'DISTRICT', 'Year', 'Murder', 'Rape',
       'Kidnapping and Abduction', 'Dacoity', 'Robbery', 'Arson', 'Hurt','Prevention of atrocities (POA) Act',
       'Protection of Civil Rights (PCR) Act',
        'Other Crimes Against SCs']]

#combining 2 CSV files

frames = [sc1 , sc13]

sc = pd.concat(frames)

sc['STATE/UT'] = sc['STATE/UT'].str.capitalize()
sc['DISTRICT'] = sc['DISTRICT'].str.capitalize()
scy = sc[sc.DISTRICT == 'Total']
scy = scy.groupby(['Year'])['Murder', 'Rape',
                            'Kidnapping and Abduction', 'Dacoity', 'Robbery', 'Arson', 'Hurt',
                            'Prevention of atrocities (POA) Act',
                            'Protection of Civil Rights (PCR) Act', 'Other Crimes Against SCs'].sum().reset_index()

crimes = ['Murder', 'Rape',
          'Kidnapping and Abduction', 'Dacoity', 'Robbery', 'Arson', 'Hurt',
          'Prevention of atrocities (POA) Act',
          'Protection of Civil Rights (PCR) Act', 'Other Crimes Against SCs']
st.subheader('# Crimes Against SC yearwise')
fig = go.Figure()
fig.add_trace(go.Scatter(x=scy['Year'], y=scy['Murder'],
                         name='Murder', line=dict(color='pink', width=4)))
fig.add_trace(go.Scatter(x=scy['Year'], y=scy['Rape'],
                         name='Rape', line=dict(color='green', width=4)))
fig.add_trace(go.Scatter(x=scy['Year'], y=scy['Kidnapping and Abduction'],
                         name='Kidnapping and Abduction', line=dict(color='orange', width=4)))
fig.add_trace(go.Scatter(x=scy['Year'], y=scy['Dacoity'],
                         name='Dacoity', line=dict(color='yellow', width=4)))
fig.add_trace(go.Scatter(x=scy['Year'], y=scy['Robbery'],
                         name='Robbery', line=dict(color='black', width=4)))
fig.add_trace(go.Scatter(x=scy['Year'], y=scy['Arson'],
                         name='Arson', line=dict(color='skyblue', width=4)))
fig.add_trace(go.Scatter(x=scy['Year'], y=scy['Hurt'],
                         name='Hurt', line=dict(color='royalblue', width=4)))
fig.add_trace(go.Scatter(x=scy['Year'], y=scy['Prevention of atrocities (POA) Act'],
                         name='Atrocities', line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x=scy['Year'], y=scy['Protection of Civil Rights (PCR) Act'],
                         mode='lines+markers',
                         name='Civil Rights Violations'))
fig.add_trace(go.Scatter(x=scy['Year'], y=scy['Other Crimes Against SCs'],
                         name='Other Crimes', line=dict(color='red', width=4)))

fig.update_layout(uniformtext_minsize=20,
                  title_text="Total Crimes Against Scs 2001-2013",

                  )

st.write(fig)

# Crime Over The Years
st.subheader('# Crime Over The Years')
scy2 = sc[sc.DISTRICT == 'Total']
scy2 = scy2.groupby(['Year'])['Murder', 'Rape',
                              'Kidnapping and Abduction', 'Dacoity', 'Robbery', 'Arson', 'Hurt',
                              'Prevention of atrocities (POA) Act',
                              'Protection of Civil Rights (PCR) Act', 'Other Crimes Against SCs'].sum().reset_index()

# Plotting Graphs
import itertools

sns.set_context("talk")
plt.style.use("fivethirtyeight")
palette = itertools.cycle(sns.color_palette("dark"))
columns = ['Murder', 'Rape',
           'Kidnapping and Abduction', 'Dacoity', 'Robbery', 'Arson', 'Hurt',
           'Prevention of atrocities (POA) Act',
           'Protection of Civil Rights (PCR) Act', 'Other Crimes Against SCs']
fig=plt.figure(figsize=(20, 30))
plt.style.use('fivethirtyeight')
for i, column in enumerate(columns):
  plt.subplot(5, 2, i + 1)
  ax = sns.barplot(data=scy2, x='Year', y=column, color=next(palette))
  plt.xlabel('')
  plt.ylabel('')
  plt.title(column, size=20)
  for p in ax.patches:
    ax.annotate("%.f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=15, color='black', xytext=(0, 8),
                textcoords='offset points')

plt.tight_layout()
plt.subplots_adjust(hspace= .3)
st.pyplot(fig)

st.subheader('Distribution of Crimes Against Scs 2001-2013')
import plotly.graph_objects as go

labels = ['Murder', 'Rape','Kidnapping', 'Dacoity', 'Robbery', 'Arson', 'Hurt','Atrocities  Act',
         'Civil Rights Act', 'Other Crimes']
values = [8576, 17991, 5305, 440,1015,2906, 54055 , 138533, 4332,176488]

fig = go.Figure(data=[go.Pie(labels=labels, values=values ,textinfo='label+percent',
                              )])
fig.update_layout(
    uniformtext_minsize= 20,
    title_text="Distribution of Crimes Against Scs 2001-2013",
    paper_bgcolor='rgb(233,233,233)',
    autosize=False,
    width=700,
    height=700)
st.write(fig)

st.subheader('Crimes Against SC StateWise')

stateyr = sc[sc.DISTRICT == 'Total']
stateyr = stateyr.groupby(['Year','STATE/UT'])['Murder', 'Rape',
       'Kidnapping and Abduction', 'Dacoity', 'Robbery', 'Arson', 'Hurt',
       'Prevention of atrocities (POA) Act',
       'Protection of Civil Rights (PCR) Act', 'Other Crimes Against SCs'].sum().reset_index()
stateyr['sum'] =  stateyr.iloc[:, 2:].sum(axis=1)
stateyr2 = stateyr.groupby('STATE/UT')['sum'].sum().reset_index()
stateyr2 = stateyr2.sort_values('sum', ascending = False)

states = ['Uttar pradesh','Rajasthan' ,'Madhya pradesh' , 'Andhra pradesh', 'Bihar', 'Karnataka' , 'Odisha' , 'Tamil nadu','Gujarat', 'Maharashtra']
sns.set_context("talk")
plt.style.use("fivethirtyeight")
fig=plt.figure(figsize = (23,28))

for i, s in enumerate(states):
    plt.subplot(5,2,i+1)
    stateyr3 = stateyr[stateyr['STATE/UT'] == s]
    ax = sns.barplot(x = 'Year' , y = 'sum' , data = stateyr3,ci=None , palette = 'colorblind' , edgecolor = 'blue')
    plt.title(s , size = 25)
    for p in ax.patches:
             ax.annotate("%.f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=15, color='black', xytext=(0, 8),
                 textcoords='offset points')
plt.tight_layout()
plt.subplots_adjust(hspace= .3)
st.pyplot(fig)

scs = sc[sc.DISTRICT == 'Total']
scs = scs.groupby(['STATE/UT'])['Murder', 'Rape',
                                'Kidnapping and Abduction', 'Dacoity', 'Robbery', 'Arson', 'Hurt',
                                'Prevention of atrocities (POA) Act',
                                'Protection of Civil Rights (PCR) Act', 'Other Crimes Against SCs'].sum().reset_index()

scs1 = scs[(scs.Murder > 100) & (scs.Rape > 100)]
sns.set_context("talk")

fig=plt.figure(figsize=(20, 30))
plt.style.use('fivethirtyeight')

for i, column in enumerate(columns):
    scs1 = scs1.sort_values(column, ascending=False)
    plt.subplot(5, 2, i + 1)
    ax = sns.barplot(data=scs1, x=column, y='STATE/UT', palette='dark')
    plt.xlabel('')
    plt.ylabel('')
    plt.title(column, size=20)
    for p in ax.patches:
        ax.annotate("%.f" % p.get_width(), xy=(p.get_width(), p.get_y() + p.get_height() / 2),
                    xytext=(5, 0), textcoords='offset points', ha="left", va="center")

plt.tight_layout()
plt.subplots_adjust(hspace=.3)
st.write(fig)

scs['sum'] = scs.sum(axis = 1)
new_row = scs.iloc[[1]]
scs = scs.append(new_row, ignore_index = True)
scs.at[35, 'STATE/UT']= 'Telangana'
scs.at[9,'STATE/UT'] = 'Nct of Delhi'


gdf = gpd.read_file("ml.zip")

gdf.st_nm = gdf.st_nm.str.lower()
scs['STATE/UT'] = scs['STATE/UT'].str.lower()

merged = gdf.merge(scs , left_on='st_nm', right_on='STATE/UT')
merged1 = merged.drop(['STATE/UT'], axis=1)

#pysal
import pysal.viz.mapclassify
import mapclassify
fig=figsize = (25, 23)
merged1['coords'] = merged1['geometry'].apply(lambda x: x.representative_point().coords[:])
merged1['coords'] = [coords[0] for coords in merged1['coords']]
colors = 8

import pylab as plot
params = {'legend.fontsize': 20,
          'legend.handlelength': 2}
plot.rcParams.update(params)


st.subheader('Crime Against SCs District Wise')
scd = sc[sc.DISTRICT != 'Total']
scd = scd.groupby(['DISTRICT'])['Murder', 'Rape',
       'Kidnapping and Abduction', 'Dacoity', 'Robbery', 'Arson', 'Hurt',
       'Prevention of atrocities (POA) Act',
       'Protection of Civil Rights (PCR) Act', 'Other Crimes Against SCs'].sum().reset_index()


sns.set_context("talk")

fig=plt.figure(figsize=(20,30))
plt.style.use('fivethirtyeight')

for i,column in enumerate(columns):
    scd1 = scd.sort_values(column,ascending = False)
    scd1 = scd1.head(10)
    plt.subplot(5,2,i+1)
    ax= sns.barplot(data= scd1,x= column ,y='DISTRICT' )
    plt.xlabel('')
    plt.ylabel('')
    plt.title(column,size = 20)
    for p in ax.patches:
        ax.annotate("%.f" % p.get_width(), xy=(p.get_width(), p.get_y()+p.get_height()/2),
            xytext=(5, 0), textcoords='offset points', ha="left", va="center")
         
    
plt.tight_layout()
plt.subplots_adjust(hspace= .3)

st.write(fig)

st.subheader('Which is the most violent District in India?')
scd['sum'] = scd['Murder']+scd['Rape']+scd['Kidnapping and Abduction']+scd['Dacoity']+scd['Robbery']+scd['Arson']+scd['Hurt']+scd['Prevention of atrocities (POA) Act']+scd['Protection of Civil Rights (PCR) Act']+scd['Other Crimes Against SCs']
mostviolent = scd.groupby(['DISTRICT'])['sum'].sum().sort_values(ascending = False).reset_index()
mostviolent = mostviolent.head(15)
import plotly.graph_objects as go


# Use textposition='auto' for direct text
fig = go.Figure(data=[go.Bar(
            x= mostviolent['DISTRICT'], y= mostviolent['sum'],
            text= mostviolent['sum'],
            textposition='auto',marker_color='rgb(255, 22, 22)'
        )])
fig.update_layout(title_text='Most Violent Districts')

st.write(fig)

st.subheader('Crime Against SCs states & District Wise')
scsd = sc[sc.DISTRICT!= 'Total']
scsd = scsd.groupby(['STATE/UT', 'DISTRICT'])['Murder', 'Rape',
       'Kidnapping and Abduction', 'Dacoity', 'Robbery', 'Arson', 'Hurt',
       'Prevention of atrocities (POA) Act',
       'Protection of Civil Rights (PCR) Act', 'Other Crimes Against SCs'].sum().reset_index()


states = ['Rajasthan', 'Maharashtra', 'Andhra pradesh', 'Uttar pradesh', 'Bihar','Madhya pradesh']
sns.set_context("talk")
sns.set_style("darkgrid")
plt.style.use('fivethirtyeight')
fig=plt.figure(figsize=(20,20))
for i , state in enumerate(states):
    scsd1 = scsd[scsd['STATE/UT'] == state].sort_values('Rape', ascending = False)
    scsd1 = scsd1.head(10)
    plt.subplot(3,2,i+1)
    ax = sns.barplot(data= scsd1,x= 'Rape' ,y= 'DISTRICT',palette = 'bright' )
    plt.xlabel('Rape Cases')
    plt.ylabel('')
    plt.title(state.capitalize(),size = 20)
    for p in ax.patches:
        ax.annotate("%.f" % p.get_width(), xy=(p.get_width(), p.get_y()+p.get_height()/2),
            xytext=(5, 0), textcoords='offset points', ha="left", va="center")
         
plt.tight_layout()
plt.subplots_adjust(hspace= .3)
st.write(fig)

