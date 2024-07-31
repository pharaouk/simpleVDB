import numpy as np
from ..utils.config import Config
from ..utils.exceptions import VectorDBException
import numpy as np
from .cluster import Cluster
from .vector import Vector
from .device_manager import DeviceManager
from ..operations import insert, delete, search, update
from ..utils.exceptions import VectorDBException
import pickle

class VectorDB:
    def __init__(self, dim, max_cluster_size=1000, cpu_memory=8e9, gpu_memory=4e9):
        self.config = Config(dim=dim, max_cluster_size=max_cluster_size)
        self.root = Cluster(id=0, centroid=np.zeros(dim))
        self.device_manager = DeviceManager(cpu_memory, gpu_memory)
        self.vector_count = 0
        self.deleted_ids = set()

    def insert(self, id, data):
        try:
            vector = Vector(id, data)
            self.root = insert.insert_vector(self.root, vector, self.config.max_cluster_size)
            self.vector_count += 1
        except Exception as e:
            raise VectorDBException(f"Insert failed: {str(e)}")

    def batch_insert(self, vectors):
        for id, data in vectors:
            self.insert(id, data)

    def delete(self, id):
        try:
            self.root = delete.delete_vector(self.root, id)
            self.vector_count -= 1
            self.deleted_ids.add(id)
        except Exception as e:
            raise VectorDBException(f"Delete failed: {str(e)}")

    def search(self, query, k=1):
        try:
            return search.search_vectors(self.root, query, k)
        except Exception as e:
            raise VectorDBException(f"Search failed: {str(e)}")

    def range_search(self, query, radius):
        try:
            return search.range_search(self.root, query, radius)
        except Exception as e:
            raise VectorDBException(f"Range search failed: {str(e)}")

    def update(self, id, new_data):
        try:
            success = update.update_vector(self.root, id, new_data)
            if not success:
                raise VectorDBException(f"Vector with id {id} not found")
        except Exception as e:
            raise VectorDBException(f"Update failed: {str(e)}")

    def optimize_memory(self):
        try:
            self.root = self.device_manager.optimize(self.root)
        except Exception as e:
            raise VectorDBException(f"Memory optimization failed: {str(e)}")


    def get_stats(self):
        return {
            "vector_count": self.vector_count,
            "deleted_ids": len(self.deleted_ids),
            "depth": self.root.depth(),
            "cpu_memory_used": self.device_manager.cpu_used,
            "gpu_memory_used": self.device_manager.gpu_used
        }
    def save(self, filepath):
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'config': self.config,
                    'root': self.root,
                    'vector_count': self.vector_count,
                    'deleted_ids': self.deleted_ids
                }, f)
        except Exception as e:
            raise VectorDBException(f"Failed to save database: {str(e)}")

    @classmethod
    def load(cls, filepath):
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            db = cls(dim=data['config'].dim, max_cluster_size=data['config'].max_cluster_size)
            db.config = data['config']
            db.root = data['root']
            db.vector_count = data['vector_count']
            db.deleted_ids = data['deleted_ids']
            return db
        except Exception as e:
            raise VectorDBException(f"Failed to load database: {str(e)}")
        

    def rebalance(self):
        try:
            self._to_cpu_recursive(self.root)
            self.root = update.rebalance_tree(self.root, self.config.max_cluster_size)
            self.optimize_memory()
        except Exception as e:
            raise VectorDBException(f"Rebalance failed: {str(e)}")

    def _to_cpu_recursive(self, node):
        if isinstance(node, Vector):
            node.to_device('cpu')
        else:
            node.to_device('cpu')
            for child in node.children:
                self._to_cpu_recursive(child)