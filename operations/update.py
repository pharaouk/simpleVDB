from ..core.vector import Vector
import numpy as np

def update_vector(node, id, new_data):
    if isinstance(node, Vector):
        if node.id == id:
            node.data = np.array(new_data)
            return True
        return False
    
    for child in node.children:
        if update_vector(child, id, new_data):
            node.update_centroid()
            return True
    
    return False

def rebalance_tree(node, max_cluster_size):
    if isinstance(node, Vector):
        return node
    
    new_children = []
    for child in node.children:
        new_child = rebalance_tree(child, max_cluster_size)
        if isinstance(new_child, list):
            new_children.extend(new_child)
        else:
            new_children.append(new_child)
    
    if len(new_children) > max_cluster_size:
        # Implement a clustering algorithm here, e.g., k-means
        # For simplicity, we'll just split into two clusters
        centroids = [child.centroid for child in new_children]
        cluster1 = Cluster(id=node.id, centroid=np.mean(centroids[:len(centroids)//2], axis=0))
        cluster2 = Cluster(id=node.id+1, centroid=np.mean(centroids[len(centroids)//2:], axis=0))
        
        for child in new_children[:len(new_children)//2]:
            cluster1.add_child(child)
        for child in new_children[len(new_children)//2:]:
            cluster2.add_child(child)
        
        return [cluster1, cluster2]
    else:
        node.children = new_children
        node.update_centroid()
        return node
    
import numpy as np
import torch
from ..core.cluster import Cluster
from ..core.vector import Vector

def update_vector(node, id, new_data):
    if isinstance(node, Vector):
        if node.id == id:
            node.data = np.array(new_data) if isinstance(new_data, (list, np.ndarray)) else new_data
            return True
        return False
    
    for child in node.children:
        if update_vector(child, id, new_data):
            node.update_centroid()
            return True
    
    return False

def rebalance_tree(node, max_cluster_size):
    if isinstance(node, Vector):
        return node
    
    new_children = []
    for child in node.children:
        new_child = rebalance_tree(child, max_cluster_size)
        if isinstance(new_child, list):
            new_children.extend(new_child)
        else:
            new_children.append(new_child)
    
    if len(new_children) > max_cluster_size:
        # Implement a clustering algorithm here, e.g., k-means
        # For simplicity, we'll just split into two clusters
        centroids = [child.centroid for child in new_children]
        if isinstance(centroids[0], np.ndarray):
            cluster1 = Cluster(id=node.id, centroid=np.mean(centroids[:len(centroids)//2], axis=0))
            cluster2 = Cluster(id=node.id+1, centroid=np.mean(centroids[len(centroids)//2:], axis=0))
        elif isinstance(centroids[0], torch.Tensor):
            cluster1 = Cluster(id=node.id, centroid=torch.mean(torch.stack(centroids[:len(centroids)//2]), dim=0))
            cluster2 = Cluster(id=node.id+1, centroid=torch.mean(torch.stack(centroids[len(centroids)//2:]), dim=0))
        else:
            raise TypeError("Unsupported centroid type")
        
        for child in new_children[:len(new_children)//2]:
            cluster1.add_child(child)
        for child in new_children[len(new_children)//2:]:
            cluster2.add_child(child)
        
        return [cluster1, cluster2]
    else:
        node.children = new_children
        node.update_centroid()
        return node