from sklearn.metrics import v_measure_score, silhouette_score
import matplotlib.pyplot as plt
from prepare_data import prepare_data
from hierarchical import hierarchical

CLUSTERS_RANGE = range(2, 10)
dataset, features, labels, num_classes = prepare_data()

v_scores = []
s_scores = []
for num_clusters in CLUSTERS_RANGE:
    predicted_clusters = hierarchical(features, num_clusters)
    v_scores.append(v_measure_score(labels, predicted_clusters))
    s_scores.append(silhouette_score(features, predicted_clusters))

v_fig = plt.figure()
v_ax = v_fig.add_subplot(1, 1, 1)
v_ax.plot(list(CLUSTERS_RANGE), v_scores)
v_ax.set_title('EXTERNAL SCORE')
v_ax.set_xlabel('Clusters amount')
v_ax.set_ylabel('V-measure score')
v_fig.savefig('./results/v_measure')

s_fig = plt.figure()
s_ax = s_fig.add_subplot(1, 1, 1)
s_ax.plot(list(CLUSTERS_RANGE), s_scores)
s_ax.set_title('INTERNAL SCORE')
s_ax.set_xlabel('Clusters amount')
s_ax.set_ylabel('Silhouette score')
s_fig.savefig('./results/silhouette')
