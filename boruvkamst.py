from typing import Dict, List, Tuple
from helper_classes import Vertex, Edge


def get_mst(V: List[Vertex], E: List[Edge]):
    # V is a set of vertices structured as an array of (x, y)
    # E is a set of edges structured as an array of ((x,y), (x,y))
    return


# FindBestNeighbors-MPC map procedure
def find_best_neighbors_map(vertex: Vertex, neighbors: List[Edge]):
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


# Contraction-MPC
def contraction_map(vertex: Vertex, neighbors: List[Vertex]) -> Tuple[Vertex, List[Edge]]:
    c = vertex # leader vertex
    s = List[Vertex]
    v = vertex # current vertex

    while s.count(v) == 0:
        s.append(v)
        v = dht_get_nearest_neighbor(v)
        if v.id < c.id: # Unsure if correct
            c = v
    
    return (c, neighbors)
    
    
def dht_get_nearest_neighbor(vertex) -> Vertex:
    # DHT stuff
    return vertex
    
