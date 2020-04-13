from sklearn.model_selection import ParameterGrid

parameters = [
    {
        'n_clusters': list(range(2, 7)),
        'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine'],
        'linkage': ['complete', 'single', 'average']
    },
    {
        'n_clusters': list(range(2, 7)),
        'affinity': ['euclidean'],
        'linkage': ['ward']
    }
]

parameter_grid = list(ParameterGrid(parameters))

best_params = {
    'n_clusters': 6,
    'affinity': 'euclidean',
    'linkage': 'complete'
}