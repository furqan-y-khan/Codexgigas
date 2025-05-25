import unittest
import os
import shutil
import json
import sys
import random
import numpy as np
from unittest.mock import patch, MagicMock, call

# --- Path Adjustments for Imports ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
if current_script_dir not in sys.path:
    sys.path.append(current_script_dir)

try:
    from code_processor import process_project_files, save_processed_data_to_json, generate_simulated_embedding
    # These are dependencies of code_processor, ensure they are importable for context
    import directory_scanner 
    import file_utils
    import vector_db_utils # Ensure this module itself is found for FAISS_AVAILABLE patching
except ImportError as e:
    print(f"Initial import failed: {e}. Attempting to add parent directory to path.")
    parent_dir = os.path.dirname(current_script_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    try:
        from code_processor import process_project_files, save_processed_data_to_json, generate_simulated_embedding
        import directory_scanner
        import file_utils
        import vector_db_utils
        print("Successfully imported from parent directory.")
    except ImportError as e_inner:
        print(f"Secondary import failed: {e_inner}. Please check your PYTHONPATH and file locations.")
        print(f"Current sys.path: {sys.path}")
        raise

# Default mocks for FAISS utility functions, can be overridden in specific tests
# These will mock the functions as imported by code_processor.py
DEFAULT_MOCK_CONFIG = {
    'code_processor.initialize_faiss_index': MagicMock(return_value=MagicMock(name="MockFaissIndex")),
    'code_processor.add_embeddings_to_index': MagicMock(),
    'code_processor.save_faiss_index': MagicMock(),
    'code_processor.search_faiss_index': MagicMock(return_value=(np.array([[0.1]]), np.array([[0]]))), # Default search result
    'code_processor.FAISS_AVAILABLE': True # Default to FAISS being available
}

class BaseTestCodeProcessor(unittest.TestCase):
    def setUp(self):
        self.test_root_dir = os.path.abspath("temp_test_code_processor_root")
        if os.path.exists(self.test_root_dir):
            shutil.rmtree(self.test_root_dir)
        os.makedirs(self.test_root_dir, exist_ok=True)
        random.seed(42)

        # Apply default mocks using patch.object or patch for the class
        self.patchers = []
        for target, mock_obj in DEFAULT_MOCK_CONFIG.items():
            # Need to handle FAISS_AVAILABLE differently as it's a boolean, not a function
            if target == 'code_processor.FAISS_AVAILABLE':
                patcher = patch(target, DEFAULT_MOCK_CONFIG[target])
            else:
                # For functions, ensure they are fresh MagicMocks for each test if setUp is per test
                patcher = patch(target, new_callable=lambda: MagicMock(return_value=mock_obj.return_value) if isinstance(mock_obj, MagicMock) else mock_obj)
            
            # If the mock_obj has a side_effect or specific configuration, apply it
            if isinstance(mock_obj, MagicMock) and hasattr(mock_obj, 'side_effect') and mock_obj.side_effect:
                patched_mock = patcher.start()
                patched_mock.side_effect = mock_obj.side_effect
            elif isinstance(mock_obj, MagicMock) and hasattr(mock_obj, 'return_value'):
                 patched_mock = patcher.start()
                 # For the main mock_index, we need to ensure it has a 'd' attribute for dimension checks
                 if target == 'code_processor.initialize_faiss_index':
                    mock_index_instance = MagicMock(name="MockFaissIndexInstance")
                    mock_index_instance.d = 0 # Default, can be set by test
                    patched_mock.return_value = mock_index_instance

            else:
                patcher.start()

            self.patchers.append(patcher)
        
        # Reset mocks that should be clean per test, especially those that count calls
        # This is tricky with class-level patching. Better to do it per method or ensure mocks are reset.
        # For simplicity here, we'll rely on method-level overrides or specific mock resets in tests.
        # Or, more simply, re-fetch the mocked objects:
        self.mock_initialize_faiss_index = DEFAULT_MOCK_CONFIG['code_processor.initialize_faiss_index']
        self.mock_add_embeddings_to_index = DEFAULT_MOCK_CONFIG['code_processor.add_embeddings_to_index']
        self.mock_save_faiss_index = DEFAULT_MOCK_CONFIG['code_processor.save_faiss_index']
        self.mock_search_faiss_index = DEFAULT_MOCK_CONFIG['code_processor.search_faiss_index']


    def tearDown(self):
        if os.path.exists(self.test_root_dir):
            shutil.rmtree(self.test_root_dir)
        for patcher in self.patchers:
            patcher.stop()
    
    def _reset_faiss_mocks(self):
        # Manually reset call counts etc. for default mocks if needed between calls in one test
        self.mock_initialize_faiss_index.reset_mock()
        self.mock_add_embeddings_to_index.reset_mock()
        self.mock_save_faiss_index.reset_mock()
        self.mock_search_faiss_index.reset_mock()


    def _create_dummy_project(self, project_name: str, files_spec: list[dict]):
        project_path = os.path.join(self.test_root_dir, project_name)
        os.makedirs(project_path, exist_ok=True)
        created_files = []
        for file_info in files_spec:
            file_path = os.path.join(project_path, file_info['name'])
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                f.write(file_info['content'])
            if file_info.get('unreadable'):
                try: os.chmod(file_path, 0o000)
                except Exception as e: print(f"Warning: Could not make {file_path} unreadable: {e}")
            created_files.append(file_path)
        return project_path, created_files

class TestCodeProcessorFaissIntegration(BaseTestCodeProcessor):

    # Test with FAISS_AVAILABLE = True (default from BaseTestCodeProcessor)
    @patch('code_processor.initialize_faiss_index', return_value=MagicMock(name="MockFaissIndexInstance", d=3))
    @patch('code_processor.add_embeddings_to_index')
    @patch('code_processor.save_faiss_index')
    @patch('code_processor.search_faiss_index', return_value=(np.array([[0.1, 0.2]]), np.array([[0,1]])))
    @patch('code_processor.FAISS_AVAILABLE', True)
    def test_content_processing_with_faiss(self, mock_faiss_available_val, mock_search, mock_save, mock_add, mock_init):
        file_content = "\n".join([f"Line {i+1}" for i in range(15)])
        files = [{'name': 'integrate.py', 'content': file_content}]
        project_path, _ = self._create_dummy_project("faiss_proj", files)
        embedding_dim = 3
        
        mock_init.return_value.d = embedding_dim # Ensure mock index has correct dimension

        results = process_project_files(project_path, ['.py'], 7, 2, embedding_dim)
        
        self.assertTrue(len(results) > 0)
        mock_init.assert_called_once_with(embedding_dim)
        
        # Check add_embeddings_to_index call
        mock_add.assert_called_once()
        args_add, _ = mock_add.call_args
        self.assertEqual(args_add[0], mock_init.return_value) # Called with the mock index
        self.assertIsInstance(args_add[1], np.ndarray)
        self.assertEqual(args_add[1].dtype, np.float32)
        self.assertEqual(args_add[1].shape[1], embedding_dim) # Check dimension of matrix

        expected_index_filename = f"{os.path.basename(project_path)}_embeddings.faiss"
        expected_index_filepath = os.path.join(project_path, expected_index_filename)
        mock_save.assert_called_once_with(mock_init.return_value, expected_index_filepath)
        
        mock_search.assert_called_once() # Assuming search demo is always run if embeddings exist


    @patch('code_processor.initialize_faiss_index')
    @patch('code_processor.add_embeddings_to_index')
    @patch('code_processor.save_faiss_index')
    @patch('code_processor.search_faiss_index')
    @patch('code_processor.FAISS_AVAILABLE', False) # Test with FAISS_AVAILABLE = False
    def test_content_processing_no_faiss(self, mock_faiss_available_val, mock_search, mock_save, mock_add, mock_init):
        file_content = "\n".join([f"Line {i+1}" for i in range(15)])
        files = [{'name': 'no_faiss_integrate.py', 'content': file_content}]
        project_path, _ = self._create_dummy_project("no_faiss_proj", files)
        
        results = process_project_files(project_path, ['.py'], 7, 2, 3)
        
        self.assertTrue(len(results) > 0) # Still processes files
        mock_init.assert_not_called()
        mock_add.assert_not_called()
        mock_save.assert_not_called()
        mock_search.assert_not_called()

    @patch('code_processor.initialize_faiss_index')
    @patch('code_processor.FAISS_AVAILABLE', True)
    def test_empty_project_skips_faiss(self, mock_faiss_available_val, mock_init):
        project_path, _ = self._create_dummy_project("empty_faiss_proj", [])
        empty_scan_dir = os.path.join(project_path, "sub")
        os.makedirs(empty_scan_dir)
        process_project_files(empty_scan_dir, ['.py'], 10, 2, 5)
        mock_init.assert_not_called() # No data to process

    @patch('code_processor.initialize_faiss_index')
    @patch('code_processor.FAISS_AVAILABLE', True)
    def test_no_matching_files_skips_faiss(self, mock_faiss_available_val, mock_init):
        files = [{'name': 'notes.txt', 'content': 'text file'}]
        project_path, _ = self._create_dummy_project("no_match_faiss_proj", files)
        process_project_files(project_path, ['.py'], 5, 1, 3) # Looking for .py, finds .txt
        mock_init.assert_not_called()

    @patch('code_processor.initialize_faiss_index')
    @patch('code_processor.FAISS_AVAILABLE', True)
    @patch('sys.stdout') # Suppress print warnings
    def test_invalid_params_skips_faiss(self, mock_stdout, mock_faiss_available_val, mock_init):
        files = [{'name': 'test.py', 'content': 'line1\nline2'}]
        project_path, _ = self._create_dummy_project("invalid_param_faiss_proj", files)
        # Invalid chunking params leading to no valid chunks/embeddings
        process_project_files(project_path, ['.py'], 0, 0, 3) 
        mock_init.assert_not_called()


    # This test focuses on the interaction with vector_db_utils.save_faiss_index
    # and the underlying (mocked) faiss.write_index call.
    @patch('code_processor.FAISS_AVAILABLE', True)
    @patch('code_processor.vector_db_utils.faiss.write_index') # Mock the actual write_index
    @patch('code_processor.initialize_faiss_index') 
    @patch('code_processor.add_embeddings_to_index')
    @patch('code_processor.search_faiss_index') # Mock search as it's part of the flow
    def test_faiss_integration_save_path_and_interaction(self, 
        mock_cp_search, mock_cp_add, mock_cp_init, mock_faiss_write_index, mock_cp_faiss_available):
        
        mock_index_instance = MagicMock(name="MockedFaissIndexForSaveTest")
        mock_cp_init.return_value = mock_index_instance

        file_content = "line1\nline2\nline3\nline4\nline5"
        files = [{'name': 'save_test.py', 'content': file_content}]
        project_path, _ = self._create_dummy_project("faiss_save_interaction_proj", files)
        embedding_dim = 3

        # We are NOT mocking code_processor.save_faiss_index itself, 
        # but the underlying faiss.write_index that it calls.
        process_project_files(project_path, ['.py'], 3, 1, embedding_dim)

        mock_cp_init.assert_called_once_with(embedding_dim)
        mock_cp_add.assert_called_once()
        
        # Assert that the actual vector_db_utils.save_faiss_index called faiss.write_index
        mock_faiss_write_index.assert_called_once()
        args_write, _ = mock_faiss_write_index.call_args
        self.assertEqual(args_write[0], mock_index_instance) # Correct index object passed
        
        expected_filename = f"{os.path.basename(project_path)}_embeddings.faiss"
        expected_filepath = os.path.join(project_path, expected_filename)
        self.assertEqual(args_write[1], expected_filepath) # Correct filepath

class TestSaveJsonFunctionality(BaseTestCodeProcessor): # Inherits setUp/tearDown
    # Test for save_processed_data_to_json (should be mostly unaffected)
    def test_successful_save_and_content_verification(self):
        random.seed(42)
        # generate_simulated_embedding is part of code_processor, so it's used directly
        sample_data = [
            {'filepath': '/path/to/file1.py', 'chunk_text': 'chunk1\nline2', 'embedding': generate_simulated_embedding("c1",3)},
            {'filepath': '/path/to/file2.py', 'chunk_text': 'chunk2', 'embedding': generate_simulated_embedding("c2",3)}
        ]
        output_json_path = os.path.join(self.test_root_dir, "output", "data.json")

        save_processed_data_to_json(sample_data, output_json_path)

        self.assertTrue(os.path.exists(output_json_path))
        with open(output_json_path, 'r') as f:
            loaded_data = json.load(f)
        self.assertEqual(loaded_data, sample_data)

    @patch('code_processor.print')
    def test_save_to_invalid_path_is_directory(self, mock_print):
        sample_data = [{'key': 'value'}]
        output_dir_path = os.path.join(self.test_root_dir, "output_is_dir_json")
        os.makedirs(output_dir_path, exist_ok=True)

        save_processed_data_to_json(sample_data, output_dir_path)
        
        error_found = False
        for call_args in mock_print.call_args_list:
            if "Error: Could not write JSON" in call_args[0][0]:
                error_found = True; break
        self.assertTrue(error_found)
        self.assertFalse(os.path.isfile(output_dir_path))

# Keep other specific tests like non_existent_project_path if they don't need FAISS specific logic
class TestCodeProcessorStandalone(BaseTestCodeProcessor):
    def test_non_existent_project_path(self):
        non_existent_path = os.path.join(self.test_root_dir, "this_project_does_not_exist")
        with self.assertRaises(FileNotFoundError): # This error is from directory_scanner
            process_project_files(non_existent_path, ['.py'], 10, 2, 5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
