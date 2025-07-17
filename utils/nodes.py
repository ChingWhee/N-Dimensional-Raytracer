import numpy as np

class Node:
    def __init__(self, coords=None):
        self.f = float('inf')
        self.g = float('inf') 
        self.h = float('inf')
        self.parent = None
        self.coords = np.array(coords) if coords is not None else np.array([])
        self.expanded = False

class Nodes:
    def __init__(self, dimensions_sizes):
        self.dimensions_sizes = np.array(dimensions_sizes)
        self.dimensions = len(self.dimensions_sizes)
        self.total_possible_nodes = np.prod(self.dimensions_sizes)
        
        self.nodes = {}
    
    def getNode(self, *coords):
        coords_tuple = tuple(coords)
        
        if len(coords_tuple) != self.dimensions:
            return None
        
        for i, coord in enumerate(coords_tuple):
            if not (0 <= coord < self.dimensions_sizes[i]):
                return None
        
        if coords_tuple not in self.nodes:
            self.nodes[coords_tuple] = Node(coords_tuple)
        
        return self.nodes[coords_tuple]
    
    def get_created_nodes_count(self):
        return len(self.nodes)
    
    def clear_unused_nodes(self):
        self.nodes = {coords: node for coords, node in self.nodes.items() 
                     if node.expanded or node.parent is not None}