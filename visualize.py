import matplotlib.pyplot as plt

class Visualize:

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