import copy
import math


# Формула Ланса-Уильямса
def dist(r, u, v, s):
    au = 0.5
    av = 0.5
    gum = 0.5
    # R(W,S) = αU ⋅ R(U,S) + αV ⋅ R(V,S) + β ⋅ R(U,V) + γ ⋅ |R(U,S) − R(V,S)|
    return au * r[u][s] + av * r[v][s] + gum * abs(r[u][s] - r[v][s])


def find_new_cluster(distances):
    x_min = copy.deepcopy(next(iter(distances)))
    y_min = copy.deepcopy(next(iter(distances[x_min])))
    min_dist = distances[x_min][y_min]
    for x in distances:
        for y in distances[x]:
            if x != y and distances[x][y] < min_dist:
                min_dist = distances[x][y]
                x_min = x
                y_min = y
    return x_min, y_min


def hier_step(distances):
    # находим 2 ближайших кластера, сливаем их, обновляем расстояния
    u, v = find_new_cluster(distances)
    new_cluster = set(u)
    new_cluster.update(v)
    new_cluster = frozenset(new_cluster)
    new_distances = {new_cluster: {}}
    for x in distances:
        if not x.issubset(new_cluster):
            new_distances[x] = {}
            for y in distances:
                if x != y and not y.issubset(new_cluster):
                    new_distances[x][y] = distances[x][y]
            new_distances[new_cluster][x] = dist(distances, u, v, x)
            new_distances[x][new_cluster] = dist(distances, u, v, x)
    return new_distances


def hierarchical(X, clusters_num):
    # расстояния между точками (евклидовы)
    distances = get_distances(X)
    it = 0
    # объединяем
    while len(distances) > clusters_num:
        it += 1
        distances = hier_step(distances)
    it = 0
    point_to_cluster = {}
    for cluster in distances.keys():
        for point in cluster:
            point_to_cluster[point] = it
        it += 1
    pred_y = [point_to_cluster[i] for i in range(len(X))]
    return pred_y


def get_distances(ds):
    distances = {}
    for i in range(len(ds)):
        distances[frozenset([i])] = {}
    for i in range(len(ds)):
        i_set = frozenset([i])
        for j in range(i + 1, len(ds)):
            j_set = frozenset([j])
            cur_dist = euclidean(ds[i], ds[j])
            distances[i_set][j_set] = cur_dist
            distances[j_set][i_set] = cur_dist
    return distances


def euclidean(x, y):
    res = 0.0
    for a, b in zip(x, y):
        res = res + (a - b) ** 2
    return math.sqrt(res)
