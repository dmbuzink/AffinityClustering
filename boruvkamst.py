from typing import Dict, List, Tuple
from helper_classes import Vertex, Edge
from pyspark import SparkConf, SparkContext, RDD


def get_mst(V: List[Vertex], E: List[Edge]):
    # V is a set of vertices structured as an array of (x, y)
    # E is a set of edges structured as an array of ((x,y), (x,y))

    data = []

    sparkConf = SparkConf().setAppName('BoruvkaMST')
    sparkContext = SparkContext(conf=sparkConf)
    
    sparkContext
    
    distData = sparkContext.parallelize(data)
    distData.map()

    return

# FindBestNeighbors-MPC implementation using PySpark
def find_best_neighbors_mpc(V: List[Vertex], E: List[Edge]) -> Tuple[Vertex, Vertex]:
    sparkConf = SparkConf().setAppName('BoruvkaMST_FindBestNeighbors')
    sparkContext = SparkContext(conf=sparkConf)
    data = []
    map_rdd = sparkContext.parallelize(data).map(lambda x: find_best_neighbors_map(x)) # key and values?
    red_rdd = map_rdd.reduceByKey(lambda x: find_best_neighbors_red(x))
    

    return red_rdd.collect()



# FindBestNeighbors-MPC map procedure
def find_best_neighbors_map(vertex: Vertex, neighbors: List[Edge]) -> Tuple[Vertex, List[Edge]]:
    leader = vertex
    if len(neighbors) > 0:
        leader = get_minimum_weight_vertex_of_neighbors(neighbors)
    return (vertex, leader)



def get_minimum_weight_vertex_of_neighbors(neighbors: List[Edge]):
    min_edge: Edge = neighbors[0]
    for i in range(1, len(neighbors)):
        ith_neighbor: Edge = neighbors[i]
        if(ith_neighbor.weight < min_edge.weight):
            min_edge = ith_neighbor
    return min_edge


# FindBestNeighbors-MPC reduce
def find_best_neighbors_red(vertex: Vertex, closest_neighbor: Edge):
    return (vertex, closest_neighbor)


# Contraction-MPC map
def contraction_map(vertex: Vertex, neighbors: List[Edge]) -> Tuple[Vertex, List[Edge]]:
    c = vertex # leader vertex
    s = List[Vertex]
    v = vertex # current vertex

    while s.count(v) == 0:
        s.append(v)
        v = dht_get_nearest_neighbor(v)
        if v.id < c.id: # Unsure if correct
            c = v
    
    return (c, neighbors)


# Contraction-MPC red
# Current version is not yet correct! (atleast I don't think so)
def contraction_red(vertex: Vertex, neigborsLists: List[List[Edge]]) -> Tuple[Vertex, List[Edge]]:
    return (vertex, contract_vertex(neigborsLists))


def list_of_edges_contains_equal_edge(edges: List[Edge], edge: Edge):
    for i in range(0, len(edges)):
       current_edge: Edge = edge[i]
       if current_edge.equals(edge):
           return True
    return False


def get_unique_edges(edges: List[Edge]):
    unique_edges = []
    for i in range(len(edges)):
        current_edge = edges[i]
        if not list_of_edges_contains_equal_edge(unique_edges, current_edge):
            unique_edges.append(current_edge)
    return edges


def flat_map_edges(list_of_edges: List[List[Edge]]):
    for i in range(len(list_of_edges)):
        edges = list_of_edges[i]
        for j in range(len(edges)):
            yield j


def contract_vertex(primary_vertex: Vertex, edges: List[Edge]) -> List[Edge]:
    leader = dht_get_leader_of_vertex(primary_vertex)

    directly_connected_vertices = []
    for i in range(len(edges)):
        current_edge = edges[i]
        if current_edge.contains_vertex(primary_vertex):
            vertex_to_add = current_edge.get_other_Vertex(primary_vertex)
            if dht_get_leader_of_vertex(vertex_to_add).equals(leader):
                directly_connected_vertices.append(vertex_to_add)

    if len(directly_connected_vertices) == 0:
        return edges

    new_edges = []
    for i in range(len(edges)):
        current_edge = edges[i]
        for j in range(len(directly_connected_vertices)):
            dc_vertex = directly_connected_vertices[j]
            # primary_vertex -> dc_vertex -> (weight W) other vertex
            # store new edge: primary_vertex -> other vertex | with weight W
            if current_edge.contains_vertex(dc_vertex) and not current_edge.contains_vertex(primary_vertex):
                new_edges.append(Edge(start_node=primary_vertex, end_node=current_edge.get_other_Vertex(primary_vertex), weight=current_edge.weight))
            # primary_vertex -> other vertex
            # keep edge
            elif not current_edge.contains_vertex(dc_vertex):
                new_edges.append(current_edge)
            # Else
            # primary_vertex -> dc_vertex -> primary_vertex
            # remove edge
    
    return contract_vertex(primary_vertex, new_edges)


nearest_neighbor_dict = {}

def dht_get_nearest_neighbor(vertex) -> Vertex:
    return nearest_neighbor_dict[vertex.id]

def dht_add_nearest_neighbor(vertex: Vertex, nearest_neighbour: Vertex):
    nearest_neighbor_dict[vertex.id] = nearest_neighbour

leader_dict = {}

def dht_get_leader_of_vertex(vertex: Vertex) -> Vertex:
    return leader_dict[vertex.id]
    
def dht_add_leader_of_vertex(key_vertex: Vertex, leader_vertex: Vertex):
    leader_dict[key_vertex.id] = leader_vertex
