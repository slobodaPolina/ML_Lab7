import numpy as np


def plot_clusters(num_clusters, values, labels, plt, filename):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)

    cm = plt.get_cmap('gist_rainbow')
    ax.set_color_cycle([cm(1. * i / num_clusters) for i in range(num_clusters)])

    for cluster_num in range(num_clusters):
        cluster_points = []

        for i in range(len(labels)):
            if labels[i] == cluster_num:
                cluster_points.append(values[i])

        cluster_points = np.array(cluster_points)

        ax.scatter(cluster_points[:, 0], cluster_points[:, 1])

    fig.savefig('./results/{}'.format(filename))