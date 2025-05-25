import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import os
import sys

# --- Path Adjustments for Imports ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming vector_db_utils.py is in the same directory or parent (project root)
if current_script_dir not in sys.path:
    sys.path.append(current_script_dir)

# Try to import target functions and their dependencies
try:
    import vector_db_utils
    from vector_db_utils import (
        initialize_faiss_index,
        add_embeddings_to_index,
        search_faiss_index,
        save_faiss_index,
        load_faiss_index
    )
except ImportError as e:
    print(f"Initial import failed for vector_db_utils: {e}. Attempting to add parent directory to path.")
    parent_dir = os.path.dirname(current_script_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    try:
        import vector_db_utils # Re-import to get the module object with potentially patched FAISS_AVAILABLE
        from vector_db_utils import (
            initialize_faiss_index,
            add_embeddings_to_index,
            search_faiss_index,
            save_faiss_index,
            load_faiss_index
        )
        print("Successfully imported from parent directory.")
    except ImportError as e_inner:
        print(f"Secondary import failed for vector_db_utils: {e_inner}. Please check your PYTHONPATH and file locations.")
        print(f"Current sys.path: {sys.path}")
        raise

# Helper class to simulate faiss module if it's not installed or for mocking
class MockFaiss:
    def __init__(self):
        self.IndexFlatL2 = MagicMock(return_value=MagicMock(spec=['add', 'search', 'd', 'ntotal', 'is_trained']))
        self.write_index = MagicMock()
        self.read_index = MagicMock(return_value=MagicMock(spec=['d', 'ntotal', 'is_trained']))

@patch('vector_db_utils.FAISS_AVAILABLE', False)
class TestVectorDbUtilsNoFaiss(unittest.TestCase):

    @patch('builtins.print')
    def test_initialize_index_no_faiss(self, mock_print):
        self.assertIsNone(initialize_faiss_index(10))
        mock_print.assert_any_call("Warning: FAISS library not available, cannot initialize index.")

    @patch('builtins.print')
    def test_add_embeddings_no_faiss(self, mock_print):
        add_embeddings_to_index(None, np.array([[1.0, 2.0]]))
        mock_print.assert_any_call("Warning: FAISS index not available or not initialized, cannot add embeddings.")

    @patch('builtins.print')
    def test_search_index_no_faiss(self, mock_print):
        self.assertIsNone(search_faiss_index(None, np.array([1.0, 2.0]), 1))
        mock_print.assert_any_call("Warning: FAISS index not available or not initialized, cannot perform search.")

    @patch('builtins.print')
    def test_save_index_no_faiss(self, mock_print):
        save_faiss_index(None, "dummy.index")
        mock_print.assert_any_call("Warning: FAISS index not available or not initialized, cannot save index.")

    @patch('builtins.print')
    def test_load_index_no_faiss(self, mock_print):
        self.assertIsNone(load_faiss_index("dummy.index"))
        mock_print.assert_any_call("Warning: FAISS library not available, cannot load index.")


# This class tests behavior AS IF faiss was successfully imported and FAISS_AVAILABLE is True.
# We achieve this by patching 'vector_db_utils.faiss' with our MockFaiss instance.
@patch('vector_db_utils.FAISS_AVAILABLE', True) # Ensure FAISS_AVAILABLE is True for these tests
@patch('vector_db_utils.faiss', new_callable=MockFaiss) # Mock the actual faiss module used by the utils
class TestVectorDbUtilsFaissAvailable(unittest.TestCase):

    def test_initialize_index_faiss_available(self, mock_faiss_module):
        dim = 128
        mock_index_instance = MagicMock()
        mock_faiss_module.IndexFlatL2.return_value = mock_index_instance
        
        index = initialize_faiss_index(dim)
        
        mock_faiss_module.IndexFlatL2.assert_called_once_with(dim)
        self.assertEqual(index, mock_index_instance)

    def test_add_embeddings_faiss_available(self, mock_faiss_module):
        mock_index = MagicMock(spec=['add'])
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        add_embeddings_to_index(mock_index, embeddings)
        
        mock_index.add.assert_called_once()
        called_with_arg = mock_index.add.call_args[0][0]
        self.assertIsInstance(called_with_arg, np.ndarray)
        self.assertEqual(called_with_arg.dtype, np.float32)
        np.testing.assert_array_equal(called_with_arg, embeddings.astype(np.float32))

    def test_search_index_faiss_available(self, mock_faiss_module):
        mock_index = MagicMock(spec=['search', 'd'])
        mock_index.d = 2 # Dimension of the index
        mock_index.search.return_value = (np.array([[0.1, 0.2]]), np.array([[0, 1]]))
        
        query_embedding = np.array([1.0, 2.0])
        top_k = 2
        
        D, I = search_faiss_index(mock_index, query_embedding, top_k)
        
        mock_index.search.assert_called_once()
        called_query_arg = mock_index.search.call_args[0][0]
        called_top_k_arg = mock_index.search.call_args[0][1]
        
        self.assertIsInstance(called_query_arg, np.ndarray)
        self.assertEqual(called_query_arg.dtype, np.float32)
        np.testing.assert_array_equal(called_query_arg, query_embedding.astype(np.float32).reshape(1,-1))
        self.assertEqual(called_top_k_arg, top_k)
        
        self.assertIsNotNone(D)
        self.assertIsNotNone(I)

    @patch('builtins.print')
    def test_search_index_dimension_mismatch(self, mock_print, mock_faiss_module):
        mock_index = MagicMock(spec=['search', 'd'])
        mock_index.d = 3 # Index dimension
        query_embedding = np.array([1.0, 2.0]) # Query dimension is 2
        top_k = 1
        
        result = search_faiss_index(mock_index, query_embedding, top_k)
        self.assertIsNone(result)
        mock_print.assert_any_call("Error: Query embedding dimension (2) does not match index dimension (3).")


    @patch('os.makedirs')
    @patch('os.path.exists', return_value=False) # Ensure directory doesn't "exist" for makedirs call
    def test_save_index_faiss_available(self, mock_path_exists, mock_makedirs, mock_faiss_module):
        mock_index = MagicMock()
        filepath = "test_dir/my.index"
        
        save_faiss_index(mock_index, filepath)
        
        mock_makedirs.assert_called_once_with(os.path.dirname(filepath), exist_ok=True)
        mock_faiss_module.write_index.assert_called_once_with(mock_index, filepath)

    @patch('os.path.exists', return_value=True)
    def test_load_index_faiss_available(self, mock_path_exists, mock_faiss_module):
        mock_loaded_index = MagicMock()
        mock_faiss_module.read_index.return_value = mock_loaded_index
        filepath = "my.index"
        
        index = load_faiss_index(filepath)
        
        mock_path_exists.assert_called_once_with(filepath)
        mock_faiss_module.read_index.assert_called_once_with(filepath)
        self.assertEqual(index, mock_loaded_index)

    @patch('builtins.print')
    @patch('os.path.exists', return_value=True)
    def test_load_index_handles_runtime_error(self, mock_path_exists, mock_print, mock_faiss_module):
        mock_faiss_module.read_index.side_effect = RuntimeError("Test FAISS read error")
        filepath = "my.index"
        
        index = load_faiss_index(filepath)
        
        self.assertIsNone(index)
        mock_print.assert_any_call("Error loading FAISS index from my.index: Test FAISS read error")

    @patch('builtins.print')
    @patch('os.path.exists', return_value=False)
    def test_load_index_file_not_found(self, mock_path_exists, mock_print, mock_faiss_module):
        filepath = "non_existent.index"
        index = load_faiss_index(filepath)
        self.assertIsNone(index)
        mock_path_exists.assert_called_once_with(filepath)
        mock_print.assert_any_call(f"Error: Index file not found at {filepath}")
        mock_faiss_module.read_index.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
