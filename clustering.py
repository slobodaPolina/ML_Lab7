from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import v_measure_score, silhouette_score
import matplotlib.pyplot as plt

from prepare_data import prepare_data
from plot import plot_clusters
from parameters import parameter_grid


data_csv = './data/glass.csv'

dataset, features, labels, num_classes = prepare_data(data_csv)


results = []

for configuration in parameter_grid:
    model = AgglomerativeClustering(**configuration)

    predicted_clusters = model.fit_predict(features)

    v_score = v_measure_score(labels, predicted_clusters)

    results.append({'params': configuration, 'score': v_score})

results = sorted(results, key=lambda k: k['score'], reverse=True)
best_params = results[0]['params']


model = AgglomerativeClustering(**best_params)
predicted_clusters = model.fit_predict(features)

v_score = v_measure_score(labels, predicted_clusters)
silhouette_score = silhouette_score(features, predicted_clusters)

print('V-measure score (external metric): {}'.format(v_score))
print('Silhouette score (internal metric): {}'.format(silhouette_score))


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(features)

plot_clusters(best_params['n_clusters'], principalComponents, predicted_clusters, plt, 'predicted_clusters')
plot_clusters(num_classes, principalComponents, labels, plt, 'true_clusters')

plt.show()