from ..core.vector import Vector
import numpy as np
import torch

class DeviceManager:
    def __init__(self, cpu_memory, gpu_memory):
        self.cpu_memory = cpu_memory
        self.gpu_memory = gpu_memory
        self.cpu_used = 0
        self.gpu_used = 0

    def allocate(self, size, preferred_device='cpu'):
        if preferred_device == 'gpu' and self.gpu_memory - self.gpu_used >= size:
            self.gpu_used += size
            return 'gpu'
        elif self.cpu_memory - self.cpu_used >= size:
            self.cpu_used += size
            return 'cpu'
        else:
            raise MemoryError("Not enough memory to allocate")

    def free(self, size, device):
        if device == 'gpu':
            self.gpu_used -= size
        else:
            self.cpu_used -= size

    def optimize(self, root):
        def _optimize(node, level):
            if isinstance(node, Vector):
                size = node.data.nbytes if isinstance(node.data, np.ndarray) else node.data.element_size() * node.data.nelement()
                preferred_device = 'gpu' if level < 3 else 'cpu'  # Adjust this heuristic as needed
                device = self.allocate(size, preferred_device)
                node.to_device(device)
            else:
                size = node.centroid.nbytes if isinstance(node.centroid, np.ndarray) else node.centroid.element_size() * node.centroid.nelement()
                preferred_device = 'gpu' if level < 2 else 'cpu'  # Adjust this heuristic as needed
                device = self.allocate(size, preferred_device)
                node.to_device(device)
                for child in node.children:
                    _optimize(child, level + 1)
            return node

        return _optimize(root, 0)