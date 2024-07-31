class VectorDBException(Exception):
    pass

class MemoryException(VectorDBException):
    pass

class IndexException(VectorDBException):
    pass

class OperationException(VectorDBException):
    pass

class HierarchyException(VectorDBException):
    pass