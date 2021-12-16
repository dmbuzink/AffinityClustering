from typing import List, Tuple
from networkx.classes.function import non_edges

import numpy as np
import networkx as nx
from numpy.core.fromnumeric import size

from sklearn.datasets import make_circles, make_moons, make_blobs
from matplotlib import pyplot as plt

from pyspark import SparkConf, SparkContext, RDD, Broadcast, AccumulatorParam

from helpers.graph import *

import noisegenerator as noisegen

# Store the SparkContext as a global variable
spark: SparkContext = None

class DictAccumulator(AccumulatorParam):
    """
    Custom accumulator definition which we use for building
    the mapping between vertices and leaders.
    """
    def zero(self, value: Dict) -> Dict:
        return {}
    def addInPlace(self, dict1: List, dict2: List) -> Dict:
        dict1.update(dict2)
        return dict1

def create_datasets(amount: int) -> List[Tuple[np.ndarray, np.ndarray]]:
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
    n_samples: int = amount
    datasets: List[Tuple[np.ndarray, np.ndarray]] = []

    blobs: Tuple[np.ndarray, np.ndarray] = make_blobs(n_samples=n_samples, random_state=8)

    datasets.append(blobs)

    return datasets

def combine_leader_lists(old_leaders: Dict[int, int], new_leaders: Dict[int, int]):
    if old_leaders is None:
        return new_leaders

    for (vertex, leader) in enumerate(old_leaders):
        leader_of_leader = new_leaders.get(leader)
        if leader_of_leader is not leader and leader_of_leader != leader:
            new_leaders[vertex] = leader_of_leader
    return new_leaders



def perform_clustering(G: Graph, k: int) -> Tuple[Graph, List[Dict[int, int]]]:
    """
    Computes the MST of a graph.

    Implementation of the MST-Sparse-MPC algorithm of the lecture notes.

    Parameters
    ----------

    G : The graph to compute the MST for.

    k : The number of clusters to return.

    Returns
    -------

    The MST of `G` as a `Graph`.

    The leaders of each level.

    """

    global spark

    # Initialize spark context
    sparkConf = SparkConf().setAppName('AffinityClustering')
    spark = SparkContext(conf=sparkConf)

    # Parallelize the edge list, which is enough to do calculations on the graph
    E = spark.parallelize(G.E.items())

    overall_leaders: List[Dict[int, int]] = []
    # Initially, each vertex belongs to its own cluster
    overall_leaders.append({v.index: v.index for v in G.V})
    num_leaders = len(G.V)
    combined_leader_list: Dict[int, int] = None

    while num_leaders > k:
        # Compute best neighbours for each vertex
        neighbours = find_best_neighbours(E)
        # Broadcast best neighbour mapping so it is available to all workers
        b_neighbours = spark.broadcast(neighbours)
        
        # Perform contraction of graph
        E_result, leaders = contract_graph(E, b_neighbours)
        combined_leader_list = combine_leader_lists(combined_leader_list, leaders)

        # Add leader mapping to list
        overall_leaders.append(leaders)
        num_leaders = len(set(leaders.values()))

        E = E_result

    result_edges = dict(E.collect())
    result_vertices = [G.V[i].copy() for i in list(result_edges.keys())]
    result_graph = Graph(result_vertices, result_edges)

    spark.stop()

    return result_graph, overall_leaders

def perform_clustering_alt(G: Graph, k: int) -> Tuple[Graph, Dict[int, int]]:

    global spark

    # Initialize spark context
    sparkConf = SparkConf().setAppName('AffinityClustering')
    spark = SparkContext(conf=sparkConf)

    # Parallelize the edge list, which is enough to do calculations on the graph
    E = spark.parallelize(G.E.items())

    overall_leaders: List[Dict[int, int]] = []
    # Initially, each vertex belongs to its own cluster
    overall_leaders.append({v.index: v.index for v in G.V})
    num_leaders = len(G.V)
    combined_leader_list: Dict[int, int] = None

    while num_leaders > k:
        # Compute best neighbours for each vertex
        neighbours = find_best_neighbours(E)
        # Broadcast best neighbour mapping so it is available to all workers
        b_neighbours = spark.broadcast(neighbours)
        
        # Perform contraction of graph
        E_result, leaders = contract_graph(E, b_neighbours)
        combined_leader_list = combine_leader_lists(combined_leader_list, leaders)

        # Add leader mapping to list
        overall_leaders.append(leaders)
        num_leaders = len(set(leaders.values()))

        E = E_result

    result_edges = dict(E.collect())
    result_vertices = [G.V[i].copy() for i in list(result_edges.keys())]
    result_graph = Graph(result_vertices, result_edges)

    spark.stop()

    return result_graph, leaders


def find_best_neighbours(E: RDD) -> Dict[int, int]:
    """
    Computes the best neighbours for each vertex using Pyspark.

    Parameters
    ----------

    E (`RDD[Tuple[int, Dict[int, float]]]`) : The edges of the graph.

    Returns
    -------

    The mapping {v: Λ(v), u: Λ(u), ...}.
    """

    def closest_neighbour(v: Tuple[int, Dict[int, float]]) -> Tuple[int, int]:
        index, neighbours = v
        if len(neighbours) == 0:
            # If v has no neighbours return the index of v
            return (index, index)
        else:
            # Return the index of the neighour with minimum edge-weight
            return (index, min(neighbours, key=neighbours.get))

    # Return the dictionary {v: Λ(v), u: Λ(u), ...}
    result = dict(E.map(closest_neighbour).collect())
    return result

def contract_graph(E: RDD, b_neighbours: Broadcast) -> Tuple[RDD, Dict[int, int]]:
    """
    Contracts the graph.

    Parameters
    ----------

    E (`RDD[Tuple[int, Dict[int, float]]]`) : The edges of the graph.

    b_neightbours (`Broadcast[Dict[int, int]]`) : Mapping of vertices to their best neighbour.

    Returns 
    -------

    (`RDD[Tuple[int, Dict[int, float]]]`) The edges of the contracted graph.

    (`Dict[int, int]`) The leaders calculated during this contraction.
    """

    global spark

    def mu(pair: Tuple[int, Dict[int, float]]) -> Tuple[int, Tuple[int, Dict[int, float]]]:

        vertex, neighbours = pair
        c = vertex
        v = vertex
        S = set()

        while v not in S:
            S.add(v)
            c = min(c, v)
            v = b_neighbours.value[v]

        return (c, (vertex, neighbours))
    
    def get_rho(leaders: Broadcast):
        
        def rho(pair: Tuple[int, List[Tuple[int, Dict[int, float]]]]) -> Tuple[int, Dict[int, float]]:

            # The index of the leader and the vertices in the component
            leader, component = pair
            # Create dictionary of edges in component
            edges = dict(component)
            # Gather set of vertices in this component
            vertices = {v for (v, neighbours) in component}

            
            # Delete non-leader vertices and edges of components
            for s in list(edges.keys()):
                for t in list(edges[s].keys()):
                    # We assume there are no self loops

                    # If both endpoints are in the component, delete the edge (s, t)
                    if s in vertices and t in vertices:
                        edges[s].pop(t)
                        continue
                # Delete the entry for s if it has no elements and it is not the leader
                if s != leader and len(edges[s]) == 0:
                    edges.pop(s)

            # For vertices with multiple edges to a single, other component,
            # choose the one with minimum weight
            for s in list(edges.keys()):
                for t in list(edges[s].keys()):
                    # Replace edge (s, t) by (leader(s), leader(t))
                    weight = edges[s].pop(t)
                    leader_t = leaders.value[t]
                    if leader_t in edges[leader]:
                        current = edges[leader][leader_t]
                        edges[leader][leader_t] = min(current, weight)
                    else:
                        edges[leader][leader_t] = weight
                # Delete the entry for s if it has no elements and it is not the leader
                if s != leader and len(edges[s]) == 0:
                    edges.pop(s)
            
            return (leader, edges[leader])
        
        return rho

    # Map each vertex to its leader
    leaders_rdd = E.map(mu)
    # Create mapping of vertices to leader
    leaders: Dict[int, int] = {}
    temp = leaders_rdd.collect()
    for (c, (v, n)) in temp:
        leaders[v] = c
    b_leaders = spark.broadcast(leaders)

    # Pass leader mapping to reduce step
    result = leaders_rdd.groupByKey().map(get_rho(b_leaders))

    return (result, leaders)

def get_cluster_class(G: Graph, overall_leaders: List[Dict[int, int]]) -> np.ndarray:
    """
    Calculates the class index vector for all vertices.

    Parameters
    ----------

    G : The graph.

    overall_leaders : The leaders at each level.

    Returns
    -------

    The class index vector for all vertices.
    """

    n = len(G.V)
    # Each vertex is its own leader initially
    classes = [i for i in range(n)]

    # The number of leaders on the last level
    last_leaders = list(set(overall_leaders[-1].values()))
    leader_to_class = {l: last_leaders.index(l) for l in last_leaders}

    for i in range(len(overall_leaders)):
        leaders = overall_leaders[i]
        for (v, l) in leaders.items():
            for j in range(len(classes)):
                if classes[j] == v:
                    classes[j] = l
    
    return np.array(list(map(lambda n: leader_to_class[n], classes)))

def combine_dataset_with_noise(dataset_points: np.ndarray, noise_points: List[Tuple[float, float]]) -> np.ndarray:
    raw_points: List[Tuple[float, float]] = dataset_points[0].tolist()
    raw_points.extend(noise_points)
    return np.array(raw_points)


def get_distinct_leaders(leader_list: Dict[int, int]):
    leaders = []
    for (vertex, leader) in enumerate(leader_list):
        if not leader_list.__contains__(leader):
            leaders.append(leader)
    return leaders


def main() -> None:
    datasets = create_datasets(50)
    noise_points = noisegen.generate_horizontal_line_equal_dist(25)

    prepped_dataset:np.ndenumerate = combine_dataset_with_noise(datasets[0], noise_points)

    G = Graph.create_from_points(prepped_dataset, threshold=10, noise_points=noise_points)

    # result_G, overall_leaders = perform_clustering(G, 3)
    result_G, total_leaders = perform_clustering_alt(G, 3)
    # result_y = get_cluster_class(result_G, overall_leaders)
    # result_G = G

    nx_graph = result_G.get_networkx_graph()

    data_X, data_y = datasets[0]
    fig, (ax_pts, ax_cluster) = plt.subplots(2)
    ax_pts.scatter(data_X[:, 0], data_X[:, 1], marker="o", c=data_y, s=25)
    # nx.draw(nx_graph, pos=result_G.get_node_pos_as_dict(), ax=ax_graph, with_labels=True)
    ax_cluster.scatter(prepped_dataset[:, 0], prepped_dataset[:, 1], marker="o", c=get_distinct_leaders(total_leaders), s=25)
    

    plt.show()


if __name__ == '__main__':
    # Initial call to main function
    main()