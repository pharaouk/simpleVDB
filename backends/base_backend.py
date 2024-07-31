from abc import ABC, abstractmethod

class BaseBackend(ABC):
    @abstractmethod
    def array(self, data):
        pass

    @abstractmethod
    def zeros(self, shape):
        pass

    @abstractmethod
    def add(self, a, b):
        pass

    @abstractmethod
    def multiply(self, a, b):
        pass

    @abstractmethod
    def normalize(self, data):
        pass

    @abstractmethod
    def dot(self, a, b):
        pass

    @abstractmethod
    def cosine_similarity(self, a, b):
        pass

    @abstractmethod
    def to_cpu(self, data):
        pass

    @abstractmethod
    def memory_usage(self, data):
        pass