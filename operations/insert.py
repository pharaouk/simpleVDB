from ..core.cluster import Cluster
from ..core.vector import Vector
import numpy as np

def insert_vector(node, vector, max_cluster_size):
    if isinstance(node, Vector):
        new_cluster = Cluster(id=node.id, centroid=node.data)
        new_cluster.add_child(node)
        new_cluster.add_child(vector)
        new_cluster.update_centroid()
        return new_cluster
    
    if len(node.children) < max_cluster_size:
        node.add_child(vector)
        node.update_centroid()
        return node
    
    closest_child = min(node.children, key=lambda child: np.linalg.norm(child.centroid - vector.data))
    result = insert_vector(closest_child, vector, max_cluster_size)
    
    if result != closest_child:
        node.remove_child(closest_child)
        node.add_child(result)
    
    node.update_centroid()
    return node