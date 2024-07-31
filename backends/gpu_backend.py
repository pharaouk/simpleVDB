import torch
from .base_backend import BaseBackend

class GPUBackend(BaseBackend):
    @staticmethod
    def array(data):
        return torch.tensor(data, device='cuda')

    @staticmethod
    def zeros(shape):
        return torch.zeros(shape, device='cuda')

    @staticmethod
    def add(a, b):
        return torch.add(a, b)

    @staticmethod
    def multiply(a, b):
        return torch.multiply(a, b)

    @staticmethod
    def normalize(data):
        return data / torch.norm(data, dim=1, keepdim=True)

    @staticmethod
    def dot(a, b):
        return torch.dot(a, b)

    @staticmethod
    def cosine_similarity(a, b):
        return torch.nn.functional.cosine_similarity(a, b, dim=0)

    @staticmethod
    def to_cpu(data):
        return data.cpu().numpy()

    @staticmethod
    def memory_usage(data):
        return data.element_size() * data.nelement()