from ..core.cluster import Cluster
from ..core.vector import Vector

def delete_vector(node, id):
    if isinstance(node, Vector):
        if node.id == id:
            return None
        return node
    
    new_children = []
    for child in node.children:
        result = delete_vector(child, id)
        if result is not None:
            new_children.append(result)
    
    node.children = new_children
    
    if len(node.children) == 0:
        return None
    elif len(node.children) == 1:
        return node.children[0]
    else:
        node.update_centroid()
        return node