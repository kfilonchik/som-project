import pickle
from tqdm import tqdm 
import numpy as np
import matplotlib.pyplot as plt
import pm4py
import dash
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import io
import base64
from PIL import Image
import pandas
import csv
from matplotlib import cm, colorbar
from matplotlib import colors as mpl_colors
import plotly.figure_factory as ff


feature_matrix = pandas.read_csv("XS_setting/clusters_30.csv",  delimiter=",", encoding='utf-8', quoting=csv.QUOTE_NONE, quotechar='"')
clusters_var = '30_clusters'
event_log_path = 'data/event_log_m2c.csv'

hex_x = feature_matrix['Ret_x']
hex_y = feature_matrix['Ret_y']
hex_size = feature_matrix[clusters_var]

app = dash.Dash(__name__)

fig = go.Figure(go.Scatter(
            x=hex_x,
            y=hex_y,
            mode='markers',
            marker=dict(
                size=20,
                symbol='hexagon',
                color=hex_size,
                colorscale='Viridis',
                showscale=True,
                line=dict(width=2, color='DarkSlateGrey')
            ),
            text=['Cluster: {:.2f}'.format(z) for z in hex_size]
        ))

fig.update_layout(
            title='SOM Hexagonal Clustering',
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            height=600
        )


# Layout of the app
app.layout = html.Div([
    dcc.Graph(id='hex-map', figure=fig),
        html.Img(id='process-model', style={'width': '100%', 'height': 'auto'})])

# Callback to update the process model based on clicked hexagon
@app.callback(Output('process-model', 'src'),[Input('hex-map', 'clickData')])
def display_process_model(clickData):
    if clickData:
        point = clickData['points'][0]
        x, y = point['x'], point['y']
               
        selected_cluster = feature_matrix[(feature_matrix['Ret_x'] == x) & (feature_matrix['Ret_y'] == y)]
        selected_case_ids = selected_cluster['case_id'].unique()

        event_log = pandas.read_csv(event_log_path,  delimiter=",", encoding='utf-8', quoting=csv.QUOTE_NONE, quotechar='"')

        event_log_filtered = event_log[event_log['case_id'].isin(selected_case_ids)]

                
        event_log = event_log_filtered.sort_values(by=['case_id', 'timestamp', 'SORTKEY'])

        event_log = pm4py.format_dataframe(event_log_filtered, case_id='case_id', activity_key='activity', timestamp_key='timestamp')
        dfg, start_activities, end_activities = pm4py.discover_dfg(event_log, activity_key='activity', case_id_key='case_id', timestamp_key='timestamp')
        pm4py.save_vis_dfg(dfg, start_activities, end_activities, 'dfg.png')


        image = Image.open("dfg.png")
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return f"data:image/png;base64,{img_str}"

    return None 


if __name__ == "__main__":
    app.run(debug=True)