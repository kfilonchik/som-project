import pandas
import pm4py
import re
import csv
import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm 
from train import TrainSOM
from preprocessing import PreProcessing
from sklearn.decomposition import PCA

# Create a function to plot the hexagonal grid
def plot_hex_map(som, data, som_width, som_height, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    markers = ['o', 's', 'D', 'v', '^', '<', '>']
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    for cnt, xx in enumerate(data):
        w = som.winner(xx)  # Getting the winning node
        plt.plot(w[0] + .5 * (w[1] % 2), w[1], markers[cnt % len(markers)], markerfacecolor='None',
                 markeredgecolor=colors[cnt % len(colors)], markersize=12, markeredgewidth=2)

    plt.xlim([0, som_width])
    plt.ylim([0, som_height])
    plt.title('Hexagonal Grid SOM')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    event_log = pandas.read_csv("data/event_log_m2c.csv",  delimiter=",", encoding='utf-8', quoting=csv.QUOTE_NONE, quotechar='"')
    #preprocessing = PreProcessing()
    #training = TrainSOM()
    event_log = pm4py.format_dataframe(event_log, case_id='case_id', activity_key='activity', timestamp_key='timestamp')
    #features = pm4py.extract_temporal_features_dataframe(event_log, case_id_key='case_id', activity_key='activity', timestamp_key='timestamp', resource_key='AUTOMATIC')
    #print(features)
    #features = pandas.read_csv("data/features-cases-more.csv",  delimiter=",", encoding='utf-8', quoting=csv.QUOTE_NONE, quotechar='"')

    #features = PreProcessing.extract_features(event_log)
    #feature_matrix = features.drop(columns=['case_id'])
    #f = features.drop(columns=['case_id'])
    #feature_matrix = PreProcessing.create_feature_matrix(features)
    #print(feature_matrix)
    cases = pandas.read_csv("data/cases-cluster.csv",  delimiter=",", encoding='utf-8', quoting=csv.QUOTE_NONE, quotechar='"')
    cases['case_id'] = event_log['case_id']
    #cases['case_duration'] = pm4py.get_all_case_durations(event_log, activity_key='activity', case_id_key='case_id', timestamp_key='timestamp')
    #cases['case_duration'] = cases['case_duration']#.astype(int) / 86400
    f = event_log.groupby('case_id').size().reset_index(name='num_steps_count')
    cases = cases.merge(f, on='case_id')
    
    app = TrainSOM.visualize_clusters(event_log, cases)

    app.run_server(debug=True)


'''
        # Use Random Forest to get feature importance
    # Perform PCA
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

    # Create a DataFrame for better readability
    loadings_df = pandas.DataFrame(pca_components.T, columns=['PC1', 'PC2'], index=f.columns)
    print(loadings_df)

    som_width, som_height, num_neurons, num_iterations = TrainSOM.config_som(features)
    print(som_width, som_height, num_neurons, num_iterations)

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

   