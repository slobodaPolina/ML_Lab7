from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import v_measure_score, silhouette_score
import matplotlib.pyplot as plt

from prepare_data import prepare_data
from parameters import best_params


data_csv = './data/glass.csv'

dataset, features, labels, num_classes = prepare_data(data_csv)


v_scores = []
s_scores = []
for num_clusters in range(2, 7):
    best_params['n_clusters'] = num_clusters

    model = AgglomerativeClustering(**best_params)

    predicted_clusters = model.fit_predict(features)

    v_score = v_measure_score(labels, predicted_clusters)
    s_score = silhouette_score(features, predicted_clusters)

    v_scores.append(v_score)
    s_scores.append(s_score)


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(features)

v_fig = plt.figure()
v_ax = v_fig.add_subplot(1,1,1)
v_ax.plot(list(range(2,7)), v_scores, c='b')
v_ax.set_title('V measure score')
v_ax.set_xlabel('Clusters')
v_ax.set_ylabel('V measure score')

s_fig = plt.figure()
s_ax = s_fig.add_subplot(1,1,1)
s_ax.plot(list(range(2,7)), s_scores, c='r')
s_ax.set_title('Silhouette score')
s_ax.set_xlabel('Clusters')
s_ax.set_ylabel('Silhouette score')

v_fig.savefig('./results/v_measure_score')
s_fig.savefig('./results/silhouette_score')