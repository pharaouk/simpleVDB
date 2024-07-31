# utils/config.py

class Config:
    def __init__(self, dim, max_cluster_size=1000, backend='cpu'):
        self.dim = dim
        self.max_cluster_size = max_cluster_size
        self.backend = backend

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Config has no attribute '{key}'")