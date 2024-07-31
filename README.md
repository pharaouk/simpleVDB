

test script:

```python
from simpleVDB import VectorDB
import numpy as np

# Create a new VectorDB with 128-dimensional vectors
db = VectorDB(dim=128)

# Batch insert vectors
vectors = [(i, np.random.randn(128)) for i in range(1000)]
db.batch_insert(vectors)

# Search for similar vectors
results = db.search(np.random.randn(128), k=5)
print("K-NN Search Results:", results)

# Range search
range_results = db.range_search(np.random.randn(128), radius=1.0)
print("Range Search Results:", range_results)

# Update a vector
db.update(500, np.random.randn(128))

# Delete a vector
db.delete(100)

# Optimize memory usage across devices
db.optimize_memory()

# Rebalance the tree
db.rebalance()

# Get database statistics
stats = db.get_stats()
print("Database Stats:", stats)




# Create and populate the database
db = VectorDB(dim=128)
vectors = [(i, np.random.randn(128)) for i in range(1000)]
db.batch_insert(vectors)

# Save the database
db.save("my_vector_db.pkl")

# Later, load the database
loaded_db = VectorDB.load("my_vector_db.pkl")

# Use the loaded database
results = loaded_db.search(np.random.randn(128), k=5)
print("Search Results:", results)
```


```md
simpleVDB/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── vector_db.py
│   ├── node.py
│   ├── cluster.py
│   ├── vector.py
│   └── device_manager.py
├── utils/
│   ├── __init__.py
│   ├── config.py
│   └── exceptions.py
├── operations/
│   ├── __init__.py
│   ├── insert.py
│   ├── delete.py
│   ├── search.py
│   └── update.py
├── backends/
│   ├── __init__.py
│   ├── cpu_backend.py
│   └── gpu_backend.py
└── tests/
    ├── __init__.py
    └── test_vector_db.py
```