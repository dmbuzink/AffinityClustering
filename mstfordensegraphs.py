import math
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.spatial
import sklearn

from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

from pyspark import RDD, SparkConf, SparkContext


# Snap stanford

def get_clustering_data():
    n_samples = 1500
    noisy_circles = make_circles(n_samples=n_samples, factor=.5,
                                          noise=.05)
    noisy_moons = make_moons(n_samples=n_samples, noise=.05)
    blobs = make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None

    # Anisotropicly distributed data
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = make_blobs(n_samples=n_samples,
                                 cluster_std=[1.0, 2.5, 0.5],
                                 random_state=random_state)

    plt.figure(figsize=(9 * 2 + 3, 13))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.95, wspace=.05,
                        hspace=.01)

    plot_num = 1

    default_base = {'quantile': .3,
                    'eps': .3,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': 3,
                    'min_samples': 20,
                    'xi': 0.05,
                    'min_cluster_size': 0.1}

    datasets = [
        (noisy_circles, {'damping': .77, 'preference': -240,
                         'quantile': .2, 'n_clusters': 2,
                         'min_samples': 20, 'xi': 0.25}),
        (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
        (varied, {'eps': .18, 'n_neighbors': 2,
                  'min_samples': 5, 'xi': 0.035, 'min_cluster_size': .2}),
        (aniso, {'eps': .15, 'n_neighbors': 2,
                 'min_samples': 20, 'xi': 0.1, 'min_cluster_size': .2}),
        (blobs, {}),
        (no_structure, {})]

    return datasets


def create_distance_matrix(dataset):
    x = []
    y = []
    for line in dataset:
        x.append([line[0]])
        y.append([line[1]])
    d_matrix = scipy.spatial.distance_matrix(x, y, threshold=1000000)
    dict = {}
    for i in range(len(d_matrix)):
        dict2 = {}
        for j in range(len(d_matrix[i])):
            dict2[j] = d_matrix[i][j]
        dict[i] = dict2
    return d_matrix, dict


def partion_V(vertices, k):
    U = []
    V = []
    random.shuffle(vertices)
    verticesU = vertices.copy()
    random.shuffle(vertices)
    verticesV = vertices.copy()
    for i in range(len(vertices)):
        if i < k:
            U.append({verticesU[i]})
            V.append({verticesV[i]})
        else:
            U[i % k].add(verticesU[i])
            V[i % k].add(verticesV[i])
    return U, V


def get_key(item):
    return item[2]


def find_mst(V, U, E):
    vertices = set()
    for v in V:
        vertices.add(v)
    for u in U:
        vertices.add(u)
    E = sorted(E, key=get_key)
    connected_component = set()
    mst = []
    remove_edges = []
    while len(mst) < len(vertices) - 1:
        for edge in E:
            if len(connected_component) == 0:
                connected_component.add(edge[0])
                connected_component.add(edge[1])
                mst.append(edge)
                E.remove(edge)
                break
            else:
                if edge[0] in connected_component:
                    if edge[1] in connected_component:
                        remove_edges.append(edge)
                        E.remove(edge)
                    else:
                        connected_component.add(edge[1])
                        mst.append(edge)
                        E.remove(edge)
                        break
                elif edge[1] in connected_component:
                    if edge[0] in connected_component:
                        remove_edges.append(edge)
                        E.remove(edge)
                    else:
                        connected_component.add(edge[0])
                        mst.append(edge)
                        E.remove(edge)
                        break
    return mst, remove_edges

def get_edges(U, V, E):
    edges = []
    for u in U:
        for v in V:
            if E[u][v] is not None:
                edges.append((u, v, E[u][v]))
    return edges


def reduce_edges(vertices, E, c, epsilon):
    n = len(vertices)
    k = math.ceil(n**((c - epsilon) / 2))
    U, V = partion_V(vertices, k)

    # map(lambda u: map(lambda v: find_mst(v, u), V), U)
    for i in range(len(U)):
        for j in range(len(V)):
            edges = get_edges(U[i], V[j], E)
            mst, removed_edges = find_mst(U[i], V[j], edges)
    return E


def create_mst(V, E, epsilon, m):
    n = len(V)
    c = math.log(m / n, n)
    # This works for now, but not forever
    # print(np.power(len(E), 2), np.power(n, 1 + epsilon))
    while np.power(len(E), 2) > np.power(n, 1 + epsilon):
        print("huh")
        E = reduce_edges(V, E, c, epsilon)
        c = (c - epsilon) / 2
    return


def main(machines, c, epsilon):
    parser = ArgumentParser()
    parser.add_argument('--test', help="Used for smaller dataset and testing", action="store_true")
    args = parser.parse_args()

    print("Start generating MST")
    if args.test:
        print("Test argument given")

    start_time = datetime.now()
    print("Starting time:", start_time)

    conf = SparkConf().setAppName('MST_Algorithm')
    sc = SparkContext(conf=conf)

    # create_mst()
    datasets = get_clustering_data()

    for dataset in datasets:
        timestamp = datetime.now()
        dm, E2 = create_distance_matrix(dataset[0][0])
        print("Created distance matrix in:", datetime.now() - timestamp)
        V = list(range(len(dm)))
        create_mst(V, E2, epsilon=0.1, m=machines)

        break

    # dataset = [[0, 0], [1, 1]]
    # create_distance_matrix(dataset)

    sc.stop()

if __name__ == '__main__':
    machines = 4
    c = 1/2 # 0 <= c <= 1
    epsilon = 1/8
    main(machines=machines, c=c, epsilon=epsilon)
