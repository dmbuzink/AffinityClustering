from _typeshed import Self
from typing import List


class Vertex:
    def __init__(self, id: int, x: float, y: float):
        self.id: int = id
        self.x: float = x
        self.y: float = y
        # self.neighbors: List[Edge] = list()

    # def add_neighbor(self: 'Vertex', vertex: 'Vertex', weight: float):
    #    self.neighbors.append(Edge(self, vertex, weight))

    def equals(self: 'Vertex', other: 'Vertex') -> bool:
        return self.x == other.x and self.y == other.y
        

class Edge:
    def __init__(self, start_node: Vertex, end_node: Vertex, weight: float):
        self.start_node: Vertex = start_node
        self.end_node: Vertex = end_node
        self.weight: float = weight

    def equals(self: 'Edge', other: 'Edge'):
        return (self.start_node.equals(other.start_node) and self.end_node.equals(other.end_node)) or (self.start_node.equals(other.end_node) and self.end_node.equals(other.start_node))

    def contains_vertex(self: 'Edge', vertex: Vertex):
        return self.start_node.equals(vertex) or self.end_node.equals(vertex)


# from collections import namedtuple

# Vertex = namedtuple("Vertex", "x y")
# Edge = namedtuple("Edge", "start_node end_node weight")