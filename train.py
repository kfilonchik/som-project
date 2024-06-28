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
## Import Matplotlib functions to create MiniSOM visualizations

from matplotlib.patches import RegularPolygon, Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable


class TrainSOM:

    def config_som(df):
        num_samples = df['case_id'].count()  # Replace with the actual number of samples in your dataset
        num_neurons = int(5 * np.sqrt(num_samples))

        som_width = int(np.sqrt(num_neurons))
        som_height = int(np.ceil(num_neurons / som_width))

        num_iterations = 500 * num_neurons

        return som_width, som_height, num_neurons, num_iterations

    def custom_train_som(som, data, num_iterations, som_width, som_height):
        # Function to plot intermediate results
        def plot_intermediate_results(som, data, iteration, total_iterations):
            plt.figure(figsize=(10, 10))
            markers = ['o', 's', 'D', 'v', '^', '<', '>']
            colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

            for cnt, xx in enumerate(data):
                w = som.winner(xx)  # Getting the winning node
                plt.plot(w[0] + .5 * (w[1] % 2), w[1], markers[cnt % len(markers)], markerfacecolor='None',
                        markeredgecolor=colors[cnt % len(colors)], markersize=12, markeredgewidth=2)

            plt.xlim([0, som_width])
            plt.ylim([0, som_height])
            plt.title(f'Hexagonal Grid SOM - Iteration {iteration}/{total_iterations}')
            plt.grid()
            plt.show()

        for i in tqdm(range(num_iterations)):
            rand_i = np.random.randint(len(data))
            som.update(data[rand_i], som.winner(data[rand_i]), i, num_iterations)
            if (i + 1) % (num_iterations // 2) == 0:  # Display intermediate results every 10%
                plot_intermediate_results(som, data, i + 1, num_iterations)

    def plot_component_planes(som,
                          figsize=(20,20),
                          features=None,
                          figrows=3,
                          title="SOM Visualization",
                          cmap=cm.Blues
                         ):

        xx, yy = som.get_euclidean_coordinates()
        weights = som.get_weights()
        

        fig = plt.figure(figsize=figsize, constrained_layout=False)
        subfigs = fig.subfigures(figrows, 
                                int(np.ceil(len(features)/figrows)), 
                                wspace=.05)
        


        for cpi, sf in zip(range(len(features)), subfigs.flatten()):
            
            colornorm = mpl_colors.Normalize(vmin=np.min(weights[::,cpi]), 
                                                vmax=np.max(weights[::,cpi]))

            axs = sf.subplots(1,1)
            axs.set_aspect('equal')
            axs.set_title(features[cpi], y=0.95)

            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    wy = yy[(i, j)] * np.sqrt(3) / 2
                    hexagon = RegularPolygon((xx[(i, j)], wy), 
                                        numVertices=6, 
                                        radius=.99 / np.sqrt(3),
                                        facecolor=cmap(colornorm(weights[i, j, cpi])), 
                                        alpha=1, 
                                        edgecolor=cmap(colornorm(weights[i, j, cpi]))
                                        )
                    axs.add_patch(hexagon)

            ## Remove axes for hex plot
            axs.margins(.05)
            axs.axis("off")

            ## Add colorbar
            divider = make_axes_locatable(axs)

            ax_cb = divider.append_axes("right", size="7%")

            ## Create a Mappable object
            cmap_sm = plt.cm.ScalarMappable(cmap=cmap, norm=colornorm)
            cmap_sm.set_array([])

            ## Create custom colorbar 
            cb1 = colorbar.Colorbar(ax_cb,
                                    orientation='vertical', 
                                    alpha=1,
                                    mappable=cmap_sm
                                    )
            cb1.ax.get_yaxis().labelpad = 6

            ## Add colorbar to plot
            sf.add_axes(ax_cb)

        fig.suptitle(title, fontsize='xx-large', y=1.05)

        plt.show()

        # Function to yield batches of data
    def batch_generator(data, batch_size):
        n_batches = len(data) // batch_size
        for i in range(n_batches):
            yield data[i * batch_size:(i + 1) * batch_size]
        if len(data) % batch_size != 0:
            yield data[n_batches * batch_size:]

    def save_model(model_filename, som):
        with open(model_filename, 'wb') as model_file:
            pickle.dump(som, model_file)

        print(f"Trained SOM model saved to {model_filename}")

    def visualize_clusters(event_log, feature_matrix):
        '''
        # Get the winning neurons for each data point
        winning_neurons = np.array([som.winner(x) for x in feature_matrix])

        # Create a DataFrame with the original data and their corresponding winning neurons
        df_clusters = event_log.copy()
        df_clusters['winning_neuron'] = [f'{x[0]}_{x[1]}' for x in winning_neurons]
        # Group cases by their winning neurons (clusters)
        clusters = df_clusters.groupby('winning_neuron')

        # Group cases by their winning neurons (clusters)
        #clusters = df_clusters.groupby('winning_neuron')

        # Prepare data for Plotly hexagonal map
        cluster_counts = df_clusters['winning_neuron'].value_counts().reset_index()
        cluster_counts.columns = ['winning_neuron', 'count']
        

        x_coords = [int(coord.split('_')[0]) for coord in cluster_counts['winning_neuron']]
        y_coords = [int(coord.split('_')[1]) for coord in cluster_counts['winning_neuron']]
        cluster_counts['x'] = x_coords
        cluster_counts['y'] = y_coords
'''
        # Create a hexagonal grid plot
        hex_x = feature_matrix['Ret_x']
        hex_y = feature_matrix['Ret_y']
        hex_size = feature_matrix['10_clusters']
        cluster_counts = feature_matrix['BMU'].value_counts().reset_index()
        #cluster_counts['x'] = cluster_counts['BMU'].apply(lambda x: int(x.split('_')[0]))
        #cluster_counts['y'] = cluster_counts['BMU'].apply(lambda x: int(x.split('_')[1]))


        # Assign clusters to the neurons based on BMU
        #clusters = event_log.groupby('BMU')['10_clusters'].mean().astype(int).reset_index()

        # Group cases by their BMU
        bmu_counts = feature_matrix.groupby('BMU').size().reset_index(name='count')
        bmu_duration = feature_matrix.groupby('BMU')['case_duration'].mean().reset_index() 

        mean_duration_per_bmu = feature_matrix.groupby('BMU')['case_duration'].mean().reset_index(name='mean_duration')


        gridsize = 20

        #fig, ax = plt.subplots()
        #hb = ax.hexbin(hex_x, hex_y, C=hex_size, gridsize=gridsize, cmap='viridis', reduce_C_function=np.mean)
        #plt.close(fig)  # Closehex the figure to avoid displaying it in non-interactive environments


        # Extract hexbin data
        #hexbin_data = {
       #     'x': hb.get_offsets()[:, 0],
       #     'y': hb.get_offsets()[:, 1],
       #     'z': hb.get_array()
        #}

        # Create a hexagonal heatmap in Plotly
        #hex_x = hexbin_data['x']
        #hex_y = hexbin_data['y']
        #hex_z = hexbin_data['z']

        fig = go.Figure(go.Scatter(
            x=hex_x,
            y=hex_y,
            mode='markers',
            marker=dict(
                size=hex_size * 10,
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




        '''

        fig = go.Figure(go.Scatter(
            x=hex_x,
            y=hex_y,
            mode='markers',
            marker=dict(
                size=hex_size * 5,  # Adjust the scaling factor for better visualization
                symbol='hexagon',
                color=hex_size,
                colorscale='Viridis',
                showscale=True,
                line=dict(width=2, color='DarkSlateGrey')
            ),
            text=feature_matrix['BMU']
        ))

        #fig = px.scatter(cluster_counts, x='x', y='y', size='count', hover_data=['winning_neuron'])
        #fig.update_traces(marker=dict(symbol='hexagon'))

        fig.update_layout(
        title='SOM Hexagonal Clustering',
        xaxis_title='X Coordinate',
        yaxis_title='Y Coordinate',
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        height=600
         )

        fig.show()
        '''
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
                #print(x,y)
                #selected_cluster = f'{int(x)}_{int(y)}'
                #selected_cases = df_clusters[df_clusters['winning_neuron'] == selected_cluster]
                selected_cluster = feature_matrix[(feature_matrix['Ret_x'] == x) & (feature_matrix['Ret_y'] == y)]
                #print("selected cluster_:", selected_cluster)
                # Extract the case IDs from the selected cluster

                
                selected_case_ids = selected_cluster['case_id'].unique()

                event_log = pandas.read_csv('data/event_log_m2c.csv',  delimiter=",", encoding='utf-8', quoting=csv.QUOTE_NONE, quotechar='"')
                #selected_case_ids = selected_cases['case_id'].unique()
                #print("selected cases_:", selected_case_ids)
                event_log_filtered = event_log[event_log['case_id'].isin(selected_case_ids)]

                
                event_log = event_log_filtered.sort_values(by=['case_id', 'timestamp', 'SORTKEY'])
                # Create an event log from the selected cases
                event_log = pm4py.format_dataframe(event_log_filtered, case_id='case_id', activity_key='activity', timestamp_key='timestamp')
                dfg, start_activities, end_activities = pm4py.discover_dfg(event_log, activity_key='activity', case_id_key='case_id', timestamp_key='timestamp')
                # Discover the process model using Alpha Miner
                #pm4py.view_dfg(dfg, start_activities, end_activities, format='svg')
                # Visualize the DFG using Matplotlib
                # Visualize the DFG as SVG
                pm4py.save_vis_dfg(dfg, start_activities, end_activities, 'dfg.png')

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
        return app



