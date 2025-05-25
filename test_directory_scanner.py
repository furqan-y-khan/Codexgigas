import unittest
import os
import shutil
import sys

# Add the parent directory to sys.path to allow import of directory_scanner
# This assumes the script is run from its directory or the parent directory.
# For a more robust solution, consider packaging or setting PYTHONPATH environment variable.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now we can import scan_directory. If directory_scanner.py is in the same directory,
# the sys.path modification might not be strictly needed but is good practice
# if they were in different subdirectories of a larger project structure.
# For this specific case, assuming directory_scanner.py is in the same directory as test_directory_scanner.py
# or the root of the project.
try:
    from directory_scanner import scan_directory
except ImportError:
    # Fallback if directory_scanner is in the same directory as this test script
    # and the sys.path adjustment wasn't quite right for the execution context.
    sys.path.insert(0, os.getcwd()) # Add current working directory
    try:
        from directory_scanner import scan_directory
    except ImportError as e:
        print(f"Failed to import scan_directory. Ensure directory_scanner.py is in the Python path.")
        print(f"Current sys.path: {sys.path}")
        raise e


class TestScanDirectory(unittest.TestCase):
    def setUp(self):
        """Set up a temporary directory for tests."""
        self.test_root_dir = os.path.abspath("temp_test_dir_for_scanner")
        # Clean up any old test directory first
        if os.path.exists(self.test_root_dir):
            shutil.rmtree(self.test_root_dir)
        os.makedirs(self.test_root_dir, exist_ok=True)

    def tearDown(self):
        """Clean up the temporary directory after tests."""
        if os.path.exists(self.test_root_dir):
            shutil.rmtree(self.test_root_dir)

    def test_empty_directory(self):
        """Test scanning an empty directory."""
        empty_dir_path = os.path.join(self.test_root_dir, "empty_subdir")
        os.makedirs(empty_dir_path)
        
        result = scan_directory(empty_dir_path)
        self.assertEqual(result, [], "Scanning an empty directory should return an empty list.")

    def test_directory_with_only_files(self):
        """Test scanning a directory containing only files."""
        files_dir_path = os.path.join(self.test_root_dir, "files_only_subdir")
        os.makedirs(files_dir_path)
        
        file_names = ["file1.txt", "file2.log", "file3.py"]
        expected_files = []
        for fname in file_names:
            file_path = os.path.join(files_dir_path, fname)
            with open(file_path, "w") as f:
                f.write(f"content of {fname}")
            expected_files.append(os.path.abspath(file_path))
            
        result = scan_directory(files_dir_path)
        self.assertEqual(len(result), len(expected_files), "Should find all files.")
        
        result_paths = sorted([item['path'] for item in result])
        expected_files.sort()
        
        self.assertEqual(result_paths, expected_files, "Paths of found files do not match expected.")
        for item in result:
            self.assertEqual(item['type'], 'file', f"Item {item['path']} should be type 'file'.")
            self.assertTrue(os.path.isabs(item['path']), f"Path {item['path']} should be absolute.")

    def test_directory_with_subdirectories(self):
        """Test scanning a directory with subdirectories and files."""
        nested_dir_path = os.path.join(self.test_root_dir, "nested_structure")
        os.makedirs(nested_dir_path)

        # Root level items
        with open(os.path.join(nested_dir_path, "root_file.txt"), "w") as f: f.write("root")
        
        # Subdirectory 1
        subdir1_path = os.path.join(nested_dir_path, "subdir1")
        os.makedirs(subdir1_path)
        with open(os.path.join(subdir1_path, "sub1_file1.txt"), "w") as f: f.write("s1f1")
        with open(os.path.join(subdir1_path, "sub1_file2.txt"), "w") as f: f.write("s1f2")

        # Subdirectory 2 (empty)
        subdir2_path = os.path.join(nested_dir_path, "subdir2")
        os.makedirs(subdir2_path)

        # Nested Subdirectory in subdir1
        sub_subdir_path = os.path.join(subdir1_path, "sub_subdir")
        os.makedirs(sub_subdir_path)
        with open(os.path.join(sub_subdir_path, "deep_file.doc"), "w") as f: f.write("deep")

        expected_items_map = {
            os.path.abspath(os.path.join(nested_dir_path, "root_file.txt")): "file",
            os.path.abspath(subdir1_path): "directory",
            os.path.abspath(os.path.join(subdir1_path, "sub1_file1.txt")): "file",
            os.path.abspath(os.path.join(subdir1_path, "sub1_file2.txt")): "file",
            os.path.abspath(subdir2_path): "directory",
            os.path.abspath(sub_subdir_path): "directory",
            os.path.abspath(os.path.join(sub_subdir_path, "deep_file.doc")): "file",
        }
        
        result = scan_directory(nested_dir_path)
        self.assertEqual(len(result), len(expected_items_map), "Should find all items in nested structure.")

        for item in result:
            self.assertTrue(os.path.isabs(item['path']), f"Path {item['path']} should be absolute.")
            self.assertIn(item['path'], expected_items_map, f"Found unexpected item: {item['path']}")
            self.assertEqual(item['type'], expected_items_map[item['path']],
                             f"Item {item['path']} has type {item['type']} but expected {expected_items_map[item['path']]}")
            del expected_items_map[item['path']] # Remove to check if all expected items were found

        self.assertEqual(len(expected_items_map), 0, f"Not all expected items were found. Missing: {expected_items_map.keys()}")


    def test_non_existent_path(self):
        """Test scanning a path that does not exist."""
        non_existent_path = os.path.join(self.test_root_dir, "this_does_not_exist")
        with self.assertRaisesRegex(FileNotFoundError, f"Error: Path '{os.path.abspath(non_existent_path)}' does not exist."):
            scan_directory(non_existent_path)

    def test_path_is_a_file(self):
        """Test scanning a path that is a file, not a directory."""
        file_path = os.path.join(self.test_root_dir, "iam_a_file.txt")
        with open(file_path, "w") as f:
            f.write("I am a file, not a directory.")
        
        with self.assertRaisesRegex(NotADirectoryError, f"Error: Path '{os.path.abspath(file_path)}' is not a directory."):
            scan_directory(file_path)

if __name__ == '__main__':
    # Ensure directory_scanner.py is found if it's in the same directory
    # This is a fallback if the initial sys.path manipulation isn't enough
    # depending on how unittest is invoked.
    if not any("directory_scanner" in m for m in sys.modules):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            from directory_scanner import scan_directory
        except ImportError:
             # If directory_scanner.py is in the root of the repo, and tests are in a subdir
            if os.path.basename(os.getcwd()) != parent_dir_name: # parent_dir_name would be project root
                sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
                from directory_scanner import scan_directory


    unittest.main()
