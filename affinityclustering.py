from typing import List, Tuple

import numpy as np
import networkx as nx

from sklearn.datasets import make_circles, make_moons, make_blobs
from matplotlib import pyplot as plt

from pyspark import SparkConf, SparkContext, RDD

from helpers.graph import *

# Store the SparkContext as a global variable
spark: SparkContext = None

def create_datasets() -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns a list of datasets.
    Each dataset consists of a tuple:
        (`n_samples`, 2) matrix of points in 2D space,
        a vector of size `n_samples` indicating the class of the point.

    Returns
    -------

    A list of datasets.
    """

    # The number of points in each dataset
    n_samples: int = 50
    datasets: List[Tuple[np.ndarray, np.ndarray]] = []

    blobs: Tuple[np.ndarray, np.ndarray] = make_blobs(n_samples=n_samples, random_state=8)

    datasets.append(blobs)

    return datasets

def compute_mst():
    """
    Implementation of the MST-Sparse-MPC algorithm of the lecture notes.


    """

    pass
    

def main() -> None:

    # Access the global spark variable
    global spark

    # Initialize spark context
    # sparkConf = SparkConf().setAppName('AffinityClustering')
    # spark = SparkContext(conf=sparkConf)

    datasets = create_datasets()

    graph = Graph.create_from_points(datasets[0], threshold=10)
    nx_graph = graph.get_networkx_graph()

    data_X, data_y = datasets[0]
    fig, (ax_pts, ax_graph) = plt.subplots(2)
    ax_pts.scatter(data_X[:, 0], data_X[:, 1], marker="o", c=data_y, s=25)
    nx.draw(nx_graph, pos=graph.get_node_pos_as_dict(), ax=ax_graph)

    plt.show()


if __name__ == '__main__':
    # Initial call to main function
    main()