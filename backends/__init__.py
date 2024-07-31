from .cpu_backend import CPUBackend
from .gpu_backend import GPUBackend

def get_backend(backend_type):
    if backend_type == 'cpu':
        return CPUBackend()
    elif backend_type == 'gpu':
        return GPUBackend()
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

__all__ = ['CPUBackend', 'GPUBackend', 'get_backend']