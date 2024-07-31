import numpy as np
from .node import Node
import torch

class Vector(Node):
    def __init__(self, id, data, parent=None):
        super().__init__(id, parent)
        self.data = np.array(data) if isinstance(data, (list, np.ndarray)) else data
        self.device = 'cpu'

    def size(self):
        return 1

    @property
    def centroid(self):
        return self.data

    @property
    def radius(self):
        return 0

    def depth(self):
        return 1

    def to_device(self, device):
        if device != self.device:
            if device == 'gpu':
                self.data = torch.from_numpy(self.data).cuda() if isinstance(self.data, np.ndarray) else self.data.cuda()
            else:
                self.data = self.data.cpu().numpy() if isinstance(self.data, torch.Tensor) else self.data
            self.device = device