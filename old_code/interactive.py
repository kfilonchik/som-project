import pandas
import pm4py
import re
import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm 
from matplotlib.patches import RegularPolygon, Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm, colorbar
from matplotlib.lines import Line2D
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import io
import base64
from PIL import Image

def import_data(file_path):
    

    ### Generate column names  (names will be 0, 1, 2, ..., maximum columns - 1)
    #column_names = ['CASEID','ACTION','START','EVENTNUMBER','AUTOMATIC','AUTOMATICFLAG','ACTIONCOST','ACTIONTIME','BELNR','E_ANLAGE','E_VKONT','E_ABLESETYP','E_ISTABLART','ZAEHLSTAND','FAKTURA','ZAHLUNG','ABRECHNUNGSSPERRE','MAHNSPERRE','STORNOGRUND','ABLSTAT','SORTKEY','DURATIONACTION','SELFLOOP','LOOP','ISREWORK','ISREWORKFLAG','LOAD_DATE']

    ### Read csv
    event_log = pandas.read_csv(file_path,  delimiter=",", encoding='utf-8', quoting=csv.QUOTE_NONE, quotechar='"')
    #print(event_log)
    event_log = pm4py.format_dataframe(event_log, case_id='case_id', activity_key='activity', timestamp_key='timestamp')
    
    event_log =  event_log.sort_values(by=['case_id', 'timestamp', 'SORTKEY'])
    case_durations = pm4py.get_all_case_durations(event_log, activity_key='activity', case_id_key='case_id', timestamp_key='timestamp')
    features = pandas.DataFrame()
    features['case_id'] = event_log['case_id'].unique()
    features['case_duration'] = case_durations
    process_steps_per_case = event_log.groupby('case_id').size().reset_index(name='process_steps')
    sum_reworks_per_case = event_log.groupby('case_id')['ISREWORK'].sum().reset_index(name='sum_reworks')
    manual_steps_df = event_log[event_log['AUTOMATIC'] != 1]

    manual_steps_per_case = manual_steps_df.groupby('case_id').size().reset_index(name='manual_steps')

    merged_df = features.merge(process_steps_per_case, on='case_id', how='left')\
                    .merge(sum_reworks_per_case, on='case_id', how='left')\
                    .merge(manual_steps_per_case, on='case_id', how='left')
    print(merged_df)

    merged_df['manual_steps'].fillna(0, inplace=True)

    return merged_df

df = import_data("data/event_log_m2c.csv")
df_numeric = df.drop(columns=['case_id'])

scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df_numeric)


data = df_normalized

model_filename = 'trained_som_model.pkl'
with open(model_filename, 'rb') as model_file:
    som = pickle.load(model_file)

print("Trained SOM model loaded successfully")

# Get the winning neurons for each data point
winning_neurons = np.array([som.winner(x) for x in data])

# Create a DataFrame with the original data and their corresponding winning neurons
df_clusters = df.copy()
df_clusters['winning_neuron'] = [f'{x[0]}_{x[1]}' for x in winning_neurons]

# Group cases by their winning neurons (clusters)
clusters = df_clusters.groupby('winning_neuron')

# Prepare data for Plotly hexagonal map
cluster_counts = df_clusters['winning_neuron'].value_counts().reset_index()
cluster_counts.columns = ['winning_neuron', 'count']

x_coords = [int(coord.split('_')[0]) for coord in cluster_counts['winning_neuron']]
y_coords = [int(coord.split('_')[1]) for coord in cluster_counts['winning_neuron']]
cluster_counts['x'] = x_coords
cluster_counts['y'] = y_coords

fig = px.scatter(cluster_counts, x='x', y='y', size='count', hover_data=['winning_neuron'])
fig.update_traces(marker=dict(symbol='hexagon'))

# Create the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    dcc.Graph(id='hex-map', figure=fig),
    html.Img(id='process-model', style={'width': '100%', 'height': 'auto'})
    #dcc.Graph(id='process-model')
])

# Callback to update the process model based on clicked hexagon
@app.callback(
    Output('process-model', 'src'),
    [Input('hex-map', 'clickData')]
)
def display_process_model(clickData):
    if clickData:
        point = clickData['points'][0]
        x, y = point['x'], point['y']
        selected_cluster = f'{int(x)}_{int(y)}'
        selected_cases = df_clusters[df_clusters['winning_neuron'] == selected_cluster]
        event_log = pandas.read_csv('data/event_log_m2c.csv',  delimiter=",", encoding='utf-8', quoting=csv.QUOTE_NONE, quotechar='"')
        selected_case_ids = selected_cases['case_id'].unique()
        event_log_filtered = event_log[event_log['case_id'].isin(selected_case_ids)]

        
        event_log = event_log_filtered.sort_values(by=['case_id', 'timestamp', 'SORTKEY'])
        # Create an event log from the selected cases
        event_log = pm4py.format_dataframe(event_log_filtered, case_id='case_id', activity_key='activity', timestamp_key='timestamp')
        dfg, start_activities, end_activities = pm4py.discover_performance_dfg(event_log, activity_key='activity', case_id_key='case_id', timestamp_key='timestamp')
        # Discover the process model using Alpha Miner
        #pm4py.view_dfg(dfg, start_activities, end_activities, format='svg')
         # Visualize the DFG using Matplotlib
        # Visualize the DFG as SVG
        pm4py.save_vis_performance_dfg(dfg, start_activities, end_activities, 'dfg.png')

        # Convert SVG to PNG for displaying in Dash
        image = Image.open("dfg.png")
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return f"data:image/png;base64,{img_str}"

       
        # Convert the PM4Py visualization to a Plotly figure
        #process_model_fig = go.Figure()
        # (Populate the process_model_fig with the visualization details)
        #return process_model_fig

    return None #go.Figure()

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
