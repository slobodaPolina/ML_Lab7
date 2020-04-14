from sklearn.decomposition import PCA
from sklearn.metrics import v_measure_score, silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from prepare_data import prepare_data
from hierarchical import hierarchical


def plot_clusters(num_clusters, values, labels, plt, filename):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    plt.get_cmap('gist_rainbow')

    for cluster_num in range(num_clusters):
        cluster_points = []
        for i in range(len(labels)):
            if labels[i] == cluster_num:
                cluster_points.append(values[i])
        cluster_points = np.array(cluster_points)
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1])
    fig.savefig('./results/{}'.format(filename))


dataset, features, labels, num_classes = prepare_data()
results = []
CLUSTERS_RANGE = range(2, 10)
# проверка всех значений числа кластеров - от 1 до n
for i in CLUSTERS_RANGE:
    predicted_clusters = hierarchical(features, i)
    # the harmonic mean between homogeneity and completeness:
    v_score = v_measure_score(labels, predicted_clusters)
    results.append({'clusters': i, 'score': v_score})

results = sorted(results, key=lambda k: k['score'], reverse=True)
best_clusters = results[0]['clusters']
print("BEST " + str(best_clusters))


predicted_clusters = hierarchical(features, best_clusters)
v_score = v_measure_score(labels, predicted_clusters)
# (b - a) / max(a, b). To clarify, b is the distance between a sample and the nearest cluster that the sample is not a part of. mean intra-cluster distance (a)
silhouette_score = silhouette_score(features, predicted_clusters)
print('EXTERNAL: {}'.format(v_score))
print('INTERNAL: {}'.format(silhouette_score))


pca = PCA(n_components=2)
transformed_features = pca.fit_transform(features)
plot_clusters(best_clusters, transformed_features, predicted_clusters, plt, 'predicted')
plot_clusters(num_classes, transformed_features, labels, plt, 'original')
plt.show()
