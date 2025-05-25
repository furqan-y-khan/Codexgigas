import unittest
import os
import shutil
import json
import sys
import random
from unittest.mock import patch

# --- Path Adjustments for Imports ---
# This assumes test_code_processor.py is in the same directory as code_processor.py,
# directory_scanner.py, and file_utils.py, or that these modules are installed/on PYTHONPATH.

current_script_dir = os.path.dirname(os.path.abspath(__file__))
# If your project structure has these files in the root and tests in a subfolder,
# you might need to add parent_dir to sys.path.
# For this setup, we assume they are in the same directory or accessible.
if current_script_dir not in sys.path:
    sys.path.append(current_script_dir)

# Try to import target functions and their dependencies
try:
    from code_processor import process_project_files, save_processed_data_to_json, generate_simulated_embedding
    # The following are used by code_processor, so their modules must be importable
    from directory_scanner import scan_directory
    from file_utils import read_file_content, chunk_code
except ImportError as e:
    print(f"Initial import failed: {e}. Attempting to add parent directory to path.")
    parent_dir = os.path.dirname(current_script_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    try:
        from code_processor import process_project_files, save_processed_data_to_json, generate_simulated_embedding
        from directory_scanner import scan_directory
        from file_utils import read_file_content, chunk_code
        print("Successfully imported from parent directory.")
    except ImportError as e_inner:
        print(f"Secondary import failed: {e_inner}. Please check your PYTHONPATH and file locations.")
        print(f"Current sys.path: {sys.path}")
        raise


class TestCodeProcessor(unittest.TestCase):

    def setUp(self):
        self.test_root_dir = os.path.abspath("temp_test_code_processor_root")
        if os.path.exists(self.test_root_dir):
            shutil.rmtree(self.test_root_dir)
        os.makedirs(self.test_root_dir, exist_ok=True)
        # Seed random for predictable embeddings in tests
        random.seed(42)


    def tearDown(self):
        if os.path.exists(self.test_root_dir):
            shutil.rmtree(self.test_root_dir)

    def _create_dummy_project(self, project_name: str, files_spec: list[dict]):
        """
        Helper to create a dummy project structure.
        files_spec: [{'name': 'path/to/file.py', 'content': '...', 'unreadable': True/False}, ...]
        """
        project_path = os.path.join(self.test_root_dir, project_name)
        os.makedirs(project_path, exist_ok=True)
        created_files = []
        for file_info in files_spec:
            file_path = os.path.join(project_path, file_info['name'])
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                f.write(file_info['content'])
            if file_info.get('unreadable'):
                try:
                    os.chmod(file_path, 0o000) # Remove all permissions
                except Exception as e:
                    print(f"Warning: Could not make {file_path} unreadable: {e}")
            created_files.append(file_path)
        return project_path, created_files

    # --- Tests for process_project_files ---

    def test_empty_project_directory(self):
        project_path, _ = self._create_dummy_project("empty_proj", [])
        # Create an empty actual subdirectory to scan
        empty_scan_dir = os.path.join(project_path, "actually_empty_subdir")
        os.makedirs(empty_scan_dir)

        result = process_project_files(empty_scan_dir, ['.py'], 10, 2, 5)
        self.assertEqual(result, [])

    def test_file_extension_filtering(self):
        files = [
            {'name': 'script1.py', 'content': 'print("py1")\nline2'},
            {'name': 'notes.txt', 'content': 'text file'},
            {'name': 'module/script2.py', 'content': 'import os\nprint("py2")'},
            {'name': 'README.md', 'content': '# Markdown'}
        ]
        project_path, _ = self._create_dummy_project("filter_proj", files)

        # Test for .py files
        py_results = process_project_files(project_path, ['.py'], 5, 1, 5)
        self.assertEqual(len(py_results), 2) # script1.py and script2.py (1 chunk each as content is short)
        py_filepaths = sorted([r['filepath'] for r in py_results])
        self.assertTrue(any("script1.py" in fp for fp in py_filepaths))
        self.assertTrue(any("script2.py" in fp for fp in py_filepaths))

        # Test for .txt files
        txt_results = process_project_files(project_path, ['.txt'], 5, 1, 5)
        self.assertEqual(len(txt_results), 1)
        self.assertTrue("notes.txt" in txt_results[0]['filepath'])
        
        # Test for .md files (expect empty)
        md_results = process_project_files(project_path, ['.md'], 5, 1, 5)
        self.assertEqual(len(md_results), 1) # README.md
        self.assertTrue("README.md" in md_results[0]['filepath'])


    def test_content_processing_and_integration(self):
        file_content = "\n".join([f"Line {i+1}" for i in range(15)]) # 15 lines
        files = [{'name': 'integrate.py', 'content': file_content}]
        project_path, _ = self._create_dummy_project("integration_proj", files)
        
        lines_per_chunk = 7
        overlap_lines = 2
        embedding_dim = 3
        random.seed(42) # ensure embedding is predictable for this test

        results = process_project_files(project_path, ['.py'], lines_per_chunk, overlap_lines, embedding_dim)
        
        self.assertTrue(len(results) > 0, "Should produce at least one chunk.")
        
        # Expected chunks:
        # C1: L1-L7
        # C2: L6-L12 (7-2=5, so starts at L6)
        # C3: L11-L15 (7-2=5, so starts at L11)
        expected_num_chunks = 3
        self.assertEqual(len(results), expected_num_chunks)

        for i, item in enumerate(results):
            self.assertTrue("integrate.py" in item['filepath'])
            self.assertIsInstance(item['chunk_text'], str)
            self.assertIsInstance(item['embedding'], list)
            self.assertEqual(len(item['embedding']), embedding_dim)
            self.assertTrue(all(isinstance(f, float) for f in item['embedding']))

            # Check chunk content (simplified check for first and last line)
            chunk_lines = item['chunk_text'].splitlines()
            if i == 0: # Chunk 1
                self.assertEqual(chunk_lines[0], "Line 1")
                self.assertEqual(chunk_lines[-1], "Line 7")
            elif i == 1: # Chunk 2
                self.assertEqual(chunk_lines[0], "Line 6")
                self.assertEqual(chunk_lines[-1], "Line 12")
            elif i == 2: # Chunk 3
                self.assertEqual(chunk_lines[0], "Line 11")
                self.assertEqual(chunk_lines[-1], "Line 15")


    @patch('sys.stdout') # To suppress print warnings in test output
    def test_handling_unreadable_files(self, mock_stdout):
        files = [
            {'name': 'readable.py', 'content': 'print("readable")\nline2'},
            {'name': 'unreadable.py', 'content': 'print("unreadable")\nline2_unreadable', 'unreadable': True}
        ]
        project_path, created_files = self._create_dummy_project("unreadable_proj", files)
        unreadable_file_path = next(f for f in created_files if "unreadable.py" in f)

        try:
            results = process_project_files(project_path, ['.py'], 5, 1, 3)
            self.assertEqual(len(results), 1) # Only readable.py should be processed
            self.assertTrue("readable.py" in results[0]['filepath'])
            # Check if the warning was printed for the unreadable file
            # This is an indirect check. A more robust way might involve patching 'print'
            # specifically in code_processor.py if it's used for warnings.
            # For now, we assume the warning mechanism works if the file is skipped.
        finally:
            # Ensure the unreadable file can be deleted
            if os.path.exists(unreadable_file_path):
                os.chmod(unreadable_file_path, 0o777)


    def test_non_existent_project_path(self):
        non_existent_path = os.path.join(self.test_root_dir, "this_project_does_not_exist")
        with self.assertRaises(FileNotFoundError):
            process_project_files(non_existent_path, ['.py'], 10, 2, 5)

    @patch('sys.stdout') # Suppress print warnings
    def test_invalid_chunking_parameters(self, mock_stdout):
        files = [{'name': 'test.py', 'content': 'line1\nline2\nline3\nline4\nline5'}]
        project_path, _ = self._create_dummy_project("invalid_chunk_proj", files)
        
        # lines_per_chunk = 0 should raise ValueError in chunk_code, caught in process_project_files
        results = process_project_files(project_path, ['.py'], 0, 0, 5)
        self.assertEqual(results, [], "Should return empty list as file processing fails.")

    @patch('sys.stdout') # Suppress print warnings
    def test_invalid_embedding_parameters(self, mock_stdout):
        files = [{'name': 'test.py', 'content': 'line1\nline2\nline3'}]
        project_path, _ = self._create_dummy_project("invalid_embed_proj", files)

        # embedding_dim = 0 should raise ValueError in generate_simulated_embedding
        results = process_project_files(project_path, ['.py'], 2, 1, 0)
        self.assertEqual(results, [], "Should return empty list as embedding generation fails.")

    # --- Tests for save_processed_data_to_json ---

    def test_successful_save_and_content_verification(self):
        random.seed(42) # for predictable embeddings
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

    @patch('code_processor.print') # Patch print in the code_processor module
    def test_save_to_invalid_path_is_directory(self, mock_print):
        sample_data = [{'key': 'value'}]
        # Use the root test directory itself as the "file" path
        output_dir_path = os.path.join(self.test_root_dir, "output_is_dir")
        os.makedirs(output_dir_path, exist_ok=True) # output_dir_path is now a directory

        save_processed_data_to_json(sample_data, output_dir_path)
        
        # Check if print was called with an error message
        # This relies on save_processed_data_to_json printing an error for IOError/OSError
        # The exact error message might vary by OS, so check for "Error: Could not write JSON"
        error_found = False
        for call_args in mock_print.call_args_list:
            if "Error: Could not write JSON" in call_args[0][0]:
                error_found = True
                break
        self.assertTrue(error_found, "Error message for writing to directory not printed.")
        self.assertFalse(os.path.isfile(output_dir_path), "File should not be created if path is a directory.")


if __name__ == '__main__':
    unittest.main(verbosity=2)
