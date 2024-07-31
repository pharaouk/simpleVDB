import numpy as np
from ..core.cluster import Cluster
from ..core.vector import Vector

import numpy as np
import torch

def search_vectors(node, query, k):
    if isinstance(node, Vector):
        if isinstance(node.data, np.ndarray) and isinstance(query, np.ndarray):
            distance = np.linalg.norm(node.data - query)
        elif isinstance(node.data, torch.Tensor) and isinstance(query, torch.Tensor):
            distance = torch.norm(node.data - query).item()
        else:
            raise TypeError("Incompatible types for distance calculation")
        return [(node.id, distance)]
    
    results = []
    sorted_children = sorted(node.children, key=lambda child: 
        (np.linalg.norm(child.centroid - query) if isinstance(child.centroid, np.ndarray) 
         else torch.norm(child.centroid - query).item()) - child.radius
    )
    
    for child in sorted_children:
        if len(results) < k or (np.linalg.norm(child.centroid - query) if isinstance(child.centroid, np.ndarray) 
                                else torch.norm(child.centroid - query).item()) - child.radius < results[-1][1]:
            results.extend(search_vectors(child, query, k))
            results = sorted(results, key=lambda x: x[1])[:k]
        else:
            break
    
    return results

def range_search(node, query, radius):
    if isinstance(node, Vector):
        if isinstance(node.data, np.ndarray) and isinstance(query, np.ndarray):
            distance = np.linalg.norm(node.data - query)
        elif isinstance(node.data, torch.Tensor) and isinstance(query, torch.Tensor):
            distance = torch.norm(node.data - query).item()
        else:
            raise TypeError("Incompatible types for distance calculation")
        if distance <= radius:
            return [(node.id, distance)]
        return []
    
    results = []
    for child in node.children:
        child_distance = (np.linalg.norm(child.centroid - query) if isinstance(child.centroid, np.ndarray) 
                          else torch.norm(child.centroid - query).item())
        if child_distance - child.radius <= radius:
            results.extend(range_search(child, query, radius))
    
    return results