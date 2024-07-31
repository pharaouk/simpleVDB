import numpy as np
from .base_backend import BaseBackend

class CPUBackend(BaseBackend):
    @staticmethod
    def array(data):
        return np.array(data)

    @staticmethod
    def zeros(shape):
        return np.zeros(shape)

    @staticmethod
    def add(a, b):
        return np.add(a, b)

    @staticmethod
    def multiply(a, b):
        return np.multiply(a, b)

    @staticmethod
    def normalize(data):
        return data / np.linalg.norm(data, axis=1, keepdims=True)

    @staticmethod
    def dot(a, b):
        return np.dot(a, b)

    @staticmethod
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    @staticmethod
    def to_cpu(data):
        return data

    @staticmethod
    def memory_usage(data):
        return data.nbytes