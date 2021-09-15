import math
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

from pyspark import RDD, SparkConf, SparkContext
from scipy.spatial import distance


def data_reading():
    pass


def reduce_edges(vertices, edges, c):
    pass


def create_mst(vertices, edges, epsilon):
    m = 0
    n = 0
    c = math.log(n / m, n)
    # c = np.log(n / m) / np.log(n)
    while len(edges) > np.power(n, 1 + epsilon):
        reduce_edges(vertices, edges, c)
        c = (c - epsilon) / 2
    return


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


def main():
    parser = ArgumentParser()
    parser.add_argument('--test', help="Used for smaller dataset and testing", action="store_true")
    args = parser.parse_args()

    print("Start generating MST")
    if args.test:
        print("Test argument given")

    start_time = datetime.now()
    checkpoint_time = start_time
    print("Starting time:", start_time)

    conf = SparkConf().setAppName('MST_Algorithm')
    sc = SparkContext(conf=conf)

    sc.stop()
    # create_mst()
    datasets = get_clustering_data()

    print(datasets[1])

if __name__ == '__main__':
    main()