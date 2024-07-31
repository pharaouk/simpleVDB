import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simpleVDB import VectorDB
from simpleVDB.utils.exceptions import VectorDBException

class TestVectorDB(unittest.TestCase):
    def setUp(self):
        self.db = VectorDB(dim=128)

    def test_create_and_read(self):
        self.db.create('test', 100)
        data = np.random.randn(100, 128)
        self.db.write('test', data)
        read_data = self.db.read('test')
        np.testing.assert_array_equal(data, read_data)

    def test_search(self):
        self.db.create('test', 100)
        data = np.random.randn(100, 128)
        self.db.write('test', data)
        query = np.random.randn(128)
        results = self.db.search('test', query, k=5)
        self.assertEqual(len(results), 5)

    def test_add(self):
        self.db.create('a', 100)
        self.db.create('b', 100)
        a = np.random.randn(100, 128)
        b = np.random.randn(100, 128)
        self.db.write('a', a)
        self.db.write('b', b)
        self.db.add('a', 'b', 'c')
        c = self.db.read('c')
        np.testing.assert_array_almost_equal(c, a + b)

    def test_normalize(self):
        self.db.create('test', 100)
        data = np.random.randn(100, 128)
        self.db.write('test', data)
        self.db.normalize('test')
        normalized = self.db.read('test')
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(100))

    def test_hierarchy(self):
        self.db.create_group('group1')
        self.db.create('vector1', 100)
        self.db.add_to_group('group1', 'vector1')
        group = self.db.get_group('group1')
        self.assertIn('vector1', group)

    def test_save_and_load(self):
        self.db.create('test', 100)
        data = np.random.randn(100, 128)
        self.db.write('test', data)
        self.db.save('test_db.pkl')
        
        new_db = VectorDB.load('test_db.pkl')
        loaded_data = new_db.read('test')
        np.testing.assert_array_equal(data, loaded_data)

    def test_exception_handling(self):
        with self.assertRaises(VectorDBException):
            self.db.read('non_existent')

if __name__ == '__main__':
    unittest.main()