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

xx, yy = som.get_euclidean_coordinates()
umatrix = som.distance_map()
weights = som.get_weights()
f = plt.figure(figsize=(10,10))
ax = f.add_subplot(111)

ax.set_aspect('equal')

# iteratively add hexagons
for i in range(weights.shape[0]):
    for j in range(weights.shape[1]):
        wy = yy[(i, j)] * np.sqrt(3) / 2
        hex = RegularPolygon((xx[(i, j)], wy), 
                             numVertices=6, 
                             radius=.95 / np.sqrt(3),
                             facecolor=cm.Blues(umatrix[i, j]), 
                             alpha=.4, 
                             edgecolor='gray')
        ax.add_patch(hex)

markers = ['o', '+', 'x']
colors = ['C0', 'C1', 'C2']
for cnt, x in enumerate(data):
    # getting the winner
    w = som.winner(x)
    # place a marker on the winning position for the sample xx
    wx, wy = som.convert_map_to_euclidean(w) 
    wy = wy * np.sqrt(3) / 2
    plt.plot(wx, wy, 
             markers[cnt % len(markers)], 
             markerfacecolor='None',
             markeredgecolor=colors[cnt % len(colors)],  
             markersize=12, 
             markeredgewidth=2)

xrange = np.arange(weights.shape[0])
yrange = np.arange(weights.shape[1])
plt.xticks(xrange-.5, xrange)
plt.yticks(yrange * np.sqrt(3) / 2, yrange)

divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal(size="5%", pad=0.05)    
cb1 = colorbar.ColorbarBase(ax_cb, cmap=cm.Blues, 
                            orientation='vertical', alpha=.4)
cb1.ax.get_yaxis().labelpad = 16
cb1.ax.set_ylabel('distance from neurons in the neighbourhood',
                  rotation=270, fontsize=16)
plt.gcf().add_axes(ax_cb)

legend_elements = [Line2D([0], [0], marker='o', color='C0', label='Kama',
                   markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                   Line2D([0], [0], marker='+', color='C1', label='Rosa',
                   markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                   Line2D([0], [0], marker='x', color='C2', label='Canadian',
                   markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2)]
ax.legend(handles=legend_elements, bbox_to_anchor=(0.1, 1.08), loc='upper left', 
          borderaxespad=0., ncol=3, fontsize=14)

plt.savefig('som_seed_hex.png')
plt.show()