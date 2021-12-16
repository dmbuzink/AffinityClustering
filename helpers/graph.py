from math import pi
from typing import List, Tuple, Dict
import numpy as np
import scipy.spatial
import networkx as nx

class Vertex:
    """
    A vertex in a graph.
    """

    def __init__(self, index: int, x: float, y: float) -> 'Vertex':
        """
        Creates a `Vertex`.

        Parameters
        ----------

        index : The index of the vertex in the graph.

        x : The x-coordinate of the vertex.

        y : The y-coordinate of the vertex.

        Returns
        -------

        The created vertex.
        """

        self.index: int = index
        self.x: float = x
        self.y: float = y

    def equals(self, other: 'Vertex') -> bool:
        return self.x == other.x and self.y == other.y

    def copy(self) -> 'Vertex':
        return Vertex(self.index, self.x, self.y)

class Graph:
    """
    A weighted, undirected graph.
    """

    def __init__(self, vertices: List[Vertex], edges: Dict[int, Dict[int, float]]) -> 'Graph':
        """
        Creates a `Graph`.

        Parameters
        ----------

        vertices: A list of `Vertex`s.

        edges: The edges of the graph, represented as a dict-of-dicts. E.g. `{1: {0: 5, 4: 2}}`, represents a graph with edges (1, 0) and (1, 4) of weights 5 and 2, respectively.

        Returns
        -------
        
        The created graph.
        """

        self.V = vertices
        self.E = edges

    def get_networkx_graph(self) -> nx.Graph:
        """
        Creates a `networkx.Graph` from this graph.

        Returns
        -------

        A `networkx.Graph`.
        """
        return nx.Graph(self.E)
    
    def get_node_pos_as_dict(self) -> Dict[int, Tuple[float, float]]:
        """
        Returns the positions of the vertices in a dictionary.

        Returns
        -------
        
        The positions of the vertices in a dictionary.
        """
        return {v.index: (v.x, v.y) for v in self.V}

    @staticmethod
    def create_from_points(data: Tuple[np.ndarray, np.ndarray], threshold: float = float("inf"), noise_points: List[Tuple[float, float]] = None) -> 'Graph':
        """
        Builds a graph from a set of points.

        Parameters
        ----------

        data : The dataset to build the graph for.

        threshold : The maximum distance between vertices that should have an edge.

        Returns
        -------

        The graph for the dataset.
        """
        
        points = data[0]
        # raw_points: List[Tuple[float, float]] = data[0].tolist()
        # raw_points.extend(noise_points)
        # points: np.ndarray = np.array(raw_points)
        n = len(points)

        # Create a list of vertices for every 2D point in the dataset
        vertices: List[Vertex] = [Vertex(i, points[i][0], points[i][1]) for i in range(n)]
        
        # # Add noise points to the list of vertices
        # for noise_point in enumerate(noise_points):
        #     vertices.append(Vertex(len(vertices), noise_point[0], noise_point[1]))



        # Compute the distance matrix for the point set
        distances = scipy.spatial.distance_matrix(points, points)

        # Edges in graph
        edges: Dict[int, Dict[int, float]] = {}
        for i in range(n):
            # The outgoing edges of vertex i: {destination: weight}
            outgoing: Dict[int, float] = {}
            for j in range(n):
                # Don't add self-edges
                if i == j:
                    continue
                # Don't add edge if it exceeds the threshold
                if distances[i][j] > threshold:
                    continue
                # Add edge
                outgoing[j] = distances[i][j]
            
            edges[i] = outgoing
        
        return Graph(vertices, edges)

