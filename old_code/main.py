import pandas
import pm4py
import re
import csv
import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm 
from old_code.train import TrainSOM
from preprocessing import PreProcessing
from sklearn.decomposition import PCA
from pm4py.algo.querying.llm.abstractions import log_to_fea_descr
import umap
import intrasom
from intrasom.visualization import PlotFactory
from intrasom.clustering import ClusterFactory
import json



if __name__ == "__main__":
    event_log = pandas.read_csv("data/event_log_m2c.csv",  delimiter=",", encoding='utf-8')
    #case_attributes = pandas.read_csv("data/case_attributes_m2c.csv",  delimiter=",", encoding='cp1252', quoting=csv.QUOTE_NONE, quotechar='"')
        # Create a dictionary for translation
    translation_dict = {
        'Fakturabeleg gedruckt/versandt': 'Invoice document printed/sent',
        'Ausgleich durch Ausgangszahlung ': 'Outgoing payment',
        'Zählerstand geändert': 'Meter reading changed',
        'Abrechnungssperre entfernt: manuell': 'Billing block removed: manually',
        'Vorläufiges Ende / kein Ausgleich': 'Preliminary end',
        'Kontokorrentbeleg angelegt': 'Invoice document created',
        'Abrechnungssperre entfernt: automatisch': 'Billing block removed: automatically',
        'Fakturabeleg angelegt': 'Factura document created',
        'Ausgleich durch Eingangszahlung': 'Incoming payment',
        'Ableseauftrag erstellt': 'Meter reading document created',
        'Abrechnung angelegt': 'Billing created',
        'Abrechnungssperre gesetzt: automatisch': 'Billing block set: automatically',
        'Ablesung erfasst': 'Meter recorded',
        'Ablesung plausibilisiert: manuell': 'Meter validated: manually',
        'Mahnstufe 4': 'Dunning level 4',
        'Ablesung unplausibel: manuell': 'Meter implausible: manually',
        'Kontokorrentbelegposition fällig': 'Invoice item due',
        'Ausgleich durch Sachbearbeitung': 'Clerical processing',
        'Ablesung unplausibel: automatisch': 'Reading implausible: automatically',
        'Abrechnungssperre gesetzt: manuell': 'Billing block set: manually',
        'Ausbuchung': 'Write-off',
        'Nicht ausgezahltes Guthaben': 'Unpaid credit',
        'Ablesung plausibilisiert: automatisch': 'Meter validated: automatically',
        'Mahnstufe 5': 'Dunning level 5',
        'Mahnstufe 2': 'Dunning level 2',
        'Mahnsperre': 'Dunning block',
        'Rückabwicklung': 'Reversal',
        'Mahnstufe 1': 'Dunning level 1',
        'Mahnstufe 6': 'Dunning level 6',
        'Mahnstufe 3': 'Dunning level 3',
        'Zählerstand fehlt': 'Meter reading missing'
    }

    # Replace German activities with English activities
    event_log.replace({'activity': translation_dict}, inplace=True)
    #preprocessing = PreProcessing()
    #training = TrainSOM()
    event_log = pm4py.format_dataframe(event_log, case_id='case_id', activity_key='activity', timestamp_key='timestamp')
    #features = pm4py.extract_temporal_features_dataframe(event_log, case_id_key='case_id', activity_key='activity', timestamp_key='timestamp', resource_key='AUTOMATIC')
    #print(features)
    #event_log = event_log.sort_values(by=['case_id', 'timestamp', 'SORTKEY'])
    #filtered_dataframe = pm4py.filter_variants_top_k(event_log, 30, activity_key='concept:name', timestamp_key='time:timestamp', case_id_key='case:concept:name')
    
    #dfg, start_activities, end_activities = pm4py.discover_dfg(filtered_dataframe, case_id_key='case:concept:name', activity_key='concept:name', timestamp_key='time:timestamp')
    #pm4py.view_dfg(dfg, start_activities, end_activities, format='png')
    
    features = pandas.read_csv("data/features-cases-all.csv",  delimiter=",", encoding='utf-8', quoting=csv.QUOTE_NONE, quotechar='"')
    #features = PreProcessing.extract_features(event_log, case_attributes)
    feature_matrix = PreProcessing.create_feature_matrix(features)
    #feature_matrix = features.drop(columns=['case_id'])
    f = features.drop(columns=['case_id'])

    '''
    som_width, som_height, num_neurons, num_iterations = TrainSOM.config_som(features)
    print(som_width, som_height, num_neurons, num_iterations)
    
    mapsize = (som_width, som_height)
    som = intrasom.SOMFactory.build(feature_matrix,
                                        mask=None,
                                        mapsize=mapsize,
                                        mapshape='toroid',
                                        lattice='hexa',
                                        normalization=None,
                                        initialization='random',
                                        neighborhood='gaussian',
                                        training='batch',
                                        name='Example',
                                        component_names=f.columns.to_list(),
                                        unit_names = None,
                                        sample_names=None,
                                        missing=False,
                                        save_nan_hist = False,
                                        pred_size=0)
    
    som.train(n_job=4, train_len_factor=2,
               previous_epoch = True)
    '''
    #bmus = pandas.read_parquet("Results/Example_neurons.parquet")
    #params = json.load(open("Results/params_Example.json", encoding='utf-8'))
    #som_r = intrasom.SOMFactory.load_som(data = feature_matrix,
                                       #trained_neurons = bmus,
                                       #params = params)
    
    #plot = PlotFactory(som_r)
    '''
    plot.plot_umatrix(figsize = (13,2.5),
                  hits = False,
                  title = "U-Matrix",
                  title_size = 20,
                  title_pad = 20,
                  legend_title = "Distance",
                  legend_title_size = 12,
                  legend_ticks_size = 7,
                  label_title_xy = (0,0.5),
                  save = True,
                  file_name = "umatrix",
                  file_path = '',
                  watermark_neurons=False)
    '''
    '''
    plot.plot_umatrix(figsize = (13,2.5),
                  hits = True,
                  title = "U-Matrix - Labeled Representative Samples",
                  title_size = 20,
                  title_pad = 20,
                  legend_title = "Distance",
                  legend_title_size = 12,
                  legend_ticks_size = 7,
                  label_title_xy = (0,0.5),
                  save = False,
                  file_name = "umatrix_sample_labels",
                  file_path = False,
                  watermark_neurons=False,
                  samples_label = True,
                  samples_label_index = range(18),
                  samples_label_fontsize = 8,
                 save_labels_rep = True)
    '''
    '''
    plot.component_plot(figsize = (12,2.5),
                    component_name = 2,
                    title_size = 20,
                    legend_title = "Presence",
                    legend_pad = 5,
                    legend_title_size = 12,
                    legend_ticks_size = 10,
                    label_title_xy = (0,0.7))
    '''
    #clustering = ClusterFactory(som_r)
    #clusters = clustering.kmeans(k=20)
    #cases = clustering.results_cluster(clusters)
    #cases.to_csv("data/clusters_20.csv", index=False)

    '''
    clustering.plot_kmeans(figsize = (12,5),
                       clusters = clusters,
                       title_size = 18,
                       title_pad = 20,
                       umatrix=True,
                       colormap = "gist_rainbow",
                       alfa_clust=0.5,
                       hits=True,
                       legend_text_size =7,
                       cluster_outline=False,
                       save=True,
                       file_name="cluster_gist_30")
    '''
    

    #print(feature_matrix)
    cases = pandas.read_csv("data/clusters_20.csv",  delimiter=",", encoding='utf-8', quoting=csv.QUOTE_NONE, quotechar='"')
    cases['case_id'] = features['case_id'].values
    #cases['case_duration'] = pm4py.get_all_case_durations(event_log, activity_key='activity', case_id_key='case_id', timestamp_key='timestamp')
    #cases['case_duration'] = cases['case_duration']#.astype(int) / 86400
    #f = event_log.groupby('case_id').size().reset_index(name='num_steps_count')
    #cases = cases.merge(f, on='case_id')
    app = TrainSOM.visualize_clusters(event_log, cases)
    app.run_server(host="localhost", port="8050", debug=True)
    # Perform PCA
    '''
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(feature_matrix)

    # Explained variance by each component
    explained_variance = pca.explained_variance_ratio_
    print('Explained variance by each component:', explained_variance)

    # Plot PCA results
    plt.scatter(pca_result[:, 0], pca_result[:, 1])
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA of Features')
    plt.show()
  
        # Get PCA components (loadings)
    pca_components = pca.components_
    print('PCA components (loadings):')
    print(pca_components)

    f = features.drop(columns=['case_id'])

    # Create a DataFrame for better readability
    loadings_df = pandas.DataFrame(pca_components.T, columns=['PC1', 'PC2'], index=f.columns)
    print(loadings_df)
  
    '''
    '''
    # Initialize the SOM
    som = MiniSom(x=som_width, y=som_height, input_len=feature_matrix.shape[1], sigma=1.5, learning_rate=.7, activation_distance='euclidean',
              topology='hexagonal', neighborhood_function='gaussian', random_seed=10)
    # Initializes the weights of the SOM picking random samples from data.
    som.random_weights_init(feature_matrix) 
    print(np.round(som.quantization_error(feature_matrix),4),"Starting QE")
    #som.train(feature_matrix.values, num_iterations)


    # Trains the SOM using all the vectors in data sequentially
    # minisom does not distinguish between unfolding and fine tuning phase;
    # Train SOM in batches
    #batch_size = 2  # Adjust batch size as needed
    #for batch in training.batch_generator(feature_matrix.values, batch_size):
        #som.train_batch(batch, num_iteration=len(batch))
    som.train_batch(feature_matrix, num_iterations)
    print(np.round(som.quantization_error(feature_matrix),4),"Ending QE")

    # Train the SOM with custom progress tracking
    model_filename = 'trained_som_model_1.pkl'
    #training.custom_train_som(som, feature_matrix, num_iterations)
    TrainSOM.save_model(model_filename, som)

    TrainSOM.plot_component_planes(som,
                      features=feature_matrix,
                      figsize=(15,15),
                      figrows=4,
                      title="SOM Component Planes ({}x{})".format(som_width,som)
                      
                     )
    
        # each neuron represents a cluster
    winner_coordinates = np.array([som.winner(x) for x in feature_matrix]).T
    # with np.ravel_multi_index we convert the bidimensional
    # coordinates to a monodimensional index
    cluster_index = np.ravel_multi_index(winner_coordinates, (som_width, som_height))

        # plotting the clusters using the first 2 dimentions of the data
    for c in np.unique(cluster_index):
        plt.scatter(feature_matrix[cluster_index == c, 0],
                    feature_matrix[cluster_index == c, 1], label='cluster='+str(c), alpha=.7)

    # plotting centroids
    for centroid in som.get_weights():
        plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', 
                    s=80, linewidths=35, color='k', label='centroid')
    plt.legend();
    '''

    # Load the trained SOM model from the file
    #with open(model_filename, 'rb') as model_file:
        #loaded_som = pickle.load(model_file)

    #print("Trained SOM model loaded successfully")
    #print(event_log, feature_matrix)

   