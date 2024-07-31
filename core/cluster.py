import numpy as np
import torch
from .node import Node

class Cluster(Node):
    def __init__(self, id, centroid, parent=None):
        super().__init__(id, parent)
        self.centroid = np.array(centroid)
        self.children = []
        self._radius = 0

    def add_child(self, child):
        self.children.append(child)
        child.parent = self
        self._update_radius()

    def remove_child(self, child):
        self.children.remove(child)
        child.parent = None
        self._update_radius()
        
    @property
    def radius(self):
        return self._radius

    def size(self):
        return sum(child.size() for child in self.children)

    def depth(self):
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)
    def to_device(self, device):
        if device != self.device:
            if device == 'gpu':
                self.centroid = torch.from_numpy(self.centroid).cuda() if isinstance(self.centroid, np.ndarray) else self.centroid.cuda()
            else:
                self.centroid = self.centroid.cpu().numpy() if isinstance(self.centroid, torch.Tensor) else self.centroid
            self.device = device
        for child in self.children:
            child.to_device(device)
    def update_centroid(self):
        if not self.children:
            return
        child_centroids = [child.centroid for child in self.children]
        if isinstance(child_centroids[0], np.ndarray):
            self.centroid = np.mean(child_centroids, axis=0)
        elif isinstance(child_centroids[0], torch.Tensor):
            self.centroid = torch.mean(torch.stack(child_centroids), dim=0)
        else:
            raise TypeError("Unsupported centroid type")
        self._update_radius()

    def _update_radius(self):
        if not self.children:
            self._radius = 0
        else:
            if isinstance(self.centroid, np.ndarray):
                self._radius = max(np.linalg.norm(child.centroid - self.centroid) + child.radius for child in self.children)
            elif isinstance(self.centroid, torch.Tensor):
                self._radius = max((child.centroid - self.centroid).norm().item() + child.radius for child in self.children)
            else:
                raise TypeError("Unsupported centroid type")