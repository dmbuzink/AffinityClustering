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
    kept_edges = List[Edge]
    for i in range(0, len(neigborsLists)):
        neighbors: List[Edge] = neigborsLists[i]
        for j in range(0, len(neighbors)):
            current_edge = neighbors[j]
            if(current_edge.contains_vertex(vertex) and not list_of_edges_contains_equal_edge(kept_edges, current_edge)): # Contains leader and edge is not already in list (but mirrored)
                kept_edges.append(current_edge)
    return (vertex, kept_edges)


def list_of_edges_contains_equal_edge(edges: List[Edge], edge: Edge):
    for i in range(0, len(edges)):
       current_edge: Edge = edge[i]
       if current_edge.equals(edge):
           return True
    return False
    
    
def dht_get_nearest_neighbor(vertex) -> Vertex:
    # DHT stuff
    return vertex
    
