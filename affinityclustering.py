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
    n_samples: int = 150
    datasets: List[Tuple[np.ndarray, np.ndarray]] = []

    blobs: Tuple[np.ndarray, np.ndarray] = make_blobs(n_samples=n_samples, random_state=8, centers=7)

    datasets.append(blobs)

    return datasets

def merge_clusters(G: Graph, edges: Dict[int, Dict[int, int]], overall_leaders: List[Dict[int, int]], k: int) -> Tuple[Graph, List[Dict[int, int]]]:
    """
    Continuously merges the closest clusters, until we have k of them.
    """

    print("Merging.")

    # Remove the last level of leaders
    overall_leaders.pop()
    # Calculate the number of leaders on the (now) last level
    num_leaders = len(set(overall_leaders[-1].values()))
    # Merge components one-by-one till we have k of them
    while num_leaders > k:
        # Find the shortest edge
        shortest: Tuple[int, int] = None
        shortest_weight = float("inf")
        for (s, neighbours) in edges.items():
            for (t, weight) in neighbours.items():
                if weight < shortest_weight:
                    shortest_weight = weight
                    shortest = (s, t)
                    
        # Merge component of t into s where (s, t) = shortest
        # I.e. have all vertices with edges to t point to s

        # Change leader of vertices
        # Copy last level
        new_leaders = dict(overall_leaders[-1])
        new_leaders[shortest[1]] = shortest[0]
        for (v, l) in new_leaders.items():
            if l == shortest[1]:
                new_leaders[v] = shortest[0]
        # Add new leaders
        overall_leaders.append(new_leaders)
        num_leaders = len(set(new_leaders.values()))

        # Remove edge from shortest[0] to shortest[1]
        edges[shortest[0]].pop(shortest[1])

        # Delete shortest[1]
        edges.pop(shortest[1])
        
        for (v, neighbours) in edges.items():
            # Move an edge to shortest[1] from a vertex to shortest[0]
            if shortest[1] in neighbours:
                weight = neighbours.pop(shortest[1])
                # If there already was an edge from v to shortest[0]
                if shortest[0] in neighbours:
                    # Replace only if shorter
                    current = neighbours[shortest[0]]
                    neighbours[shortest[0]] = min(current, weight)
                else:
                    neighbours[shortest[0]] = weight
                # Keep edges symmetric
                edges[shortest[0]][v] = edges[v][shortest[0]]
    
    # Return graph
    result_vertices = [G.V[i].copy() for i in list(edges.keys())]
    result_graph = Graph(result_vertices, edges)

    return result_graph, overall_leaders

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
    E_prev: RDD = None
    E = spark.parallelize(G.E.items())

    overall_leaders: List[Dict[int, int]] = []
    # Initially, each vertex belongs to its own cluster
    overall_leaders.append({v.index: v.index for v in G.V})
    num_leaders = len(G.V)

    while num_leaders > k:
        # Compute best neighbours for each vertex
        neighbours = find_best_neighbours(E)
        # Broadcast best neighbour mapping so it is available to all workers
        b_neighbours = spark.broadcast(neighbours)
        
        # Perform contraction of graph
        E_result, leaders = contract_graph(E, b_neighbours)

        # Add leader mapping to list
        overall_leaders.append(leaders)
        num_leaders = len(set(leaders.values()))

        # We've gone too far, we need to "go back" to the previous step
        # and merge clusters manually until we get the desired number
        # of clusters
        if num_leaders < k:
            edges: Dict[int, Dict[int, float]] = dict(E_prev.collect())
            # Careful! We're passing overall_leaders by reference!
            return merge_clusters(G, edges, overall_leaders, k)
        else:
            # Continue to next iteration
            E_prev = E
            E = E_result

    result_edges = dict(E.collect())
    result_vertices = [G.V[i].copy() for i in list(result_edges.keys())]
    result_graph = Graph(result_vertices, result_edges)

    spark.stop()

    return result_graph, overall_leaders


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


def main() -> None:
    datasets = create_datasets()
    # noise_points = noisegen.generate_horizontal_line_equal_dist(25)

    G = Graph.create_from_points(datasets[0], threshold=10)

    result_G, overall_leaders = perform_clustering(G, 7)
    result_y = get_cluster_class(G, overall_leaders)
    # result_G = G

    nx_graph = result_G.get_networkx_graph()

    data_X, data_y = datasets[0]
    fig, (ax_pts, ax_cluster) = plt.subplots(2)
    ax_pts.scatter(data_X[:, 0], data_X[:, 1], marker="o", c=data_y, s=25)
    # nx.draw(nx_graph, pos=result_G.get_node_pos_as_dict(), ax=ax_graph, with_labels=True)
    ax_cluster.scatter(data_X[:, 0], data_X[:, 1], marker="o", c=result_y, s=25)

    plt.show()


if __name__ == '__main__':
    # Initial call to main function
    main()