from abc import ABC, abstractmethod
import pickle
class Node(ABC):
    def __init__(self, id, parent=None):
        self.id = id
        self.parent = parent
        self.device = 'cpu'  

    @abstractmethod
    def size(self):
        pass

    def to_device(self, device):
        self.device = device

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)