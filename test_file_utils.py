import unittest
import os
import shutil
import sys

# Add the parent directory to sys.path to allow import of file_utils
# This assumes the script is run from its directory or the parent directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # This would be the project root if tests are in a subdir

# Attempt to import from file_utils, adjusting path if necessary
try:
    from file_utils import read_file_content, chunk_code
except ImportError:
    # If file_utils.py is in the same directory as this test script
    sys.path.insert(0, os.getcwd())
    try:
        from file_utils import read_file_content, chunk_code
    except ImportError as e:
        # If file_utils.py is in the parent directory (project root)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        try:
            from file_utils import read_file_content, chunk_code
        except ImportError as e_inner:
            print(f"Failed to import functions from file_utils.py. Ensure it's in the Python path.")
            print(f"Current sys.path: {sys.path}")
            print(f"Error: {e_inner}")
            raise


class TestFileUtils(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory for tests."""
        self.test_dir = os.path.abspath("temp_test_file_utils_dir")
        # Clean up any old test directory first
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)

        self.dummy_file_name = "test_file.txt"
        self.dummy_file_path = os.path.join(self.test_dir, self.dummy_file_name)
        self.dummy_dir_name = "test_subdir_for_read" # Renamed to avoid conflict
        self.dummy_dir_path = os.path.join(self.test_dir, self.dummy_dir_name)


    def tearDown(self):
        """Clean up the temporary directory after tests."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    # --- Tests for read_file_content ---

    def test_read_valid_file(self):
        """Test reading content from a valid file."""
        known_content = "Hello, world!\nThis is a test file with multiple lines.\nLine 3 here."
        with open(self.dummy_file_path, "w") as f:
            f.write(known_content)
        
        content = read_file_content(self.dummy_file_path)
        self.assertEqual(content, known_content)

    def test_read_file_not_found(self):
        """Test reading a non-existent file."""
        non_existent_path = os.path.join(self.test_dir, "non_existent.txt")
        # The function returns a string, not raises an error for this case.
        # The path in the error message is not made absolute by read_file_content, so we use the direct path.
        expected_message = f"Error: File not found at {non_existent_path}"
        self.assertEqual(read_file_content(non_existent_path), expected_message)

    def test_read_path_is_directory(self):
        """Test reading a path that is a directory."""
        if not os.path.exists(self.dummy_dir_path):
            os.makedirs(self.dummy_dir_path)
        # The function returns a string, not raises an error for this case.
        # The path in the error message is not made absolute by read_file_content.
        expected_message = f"Error: Path is a directory, not a file at {self.dummy_dir_path}"
        self.assertEqual(read_file_content(self.dummy_dir_path), expected_message)

    # --- Tests for chunk_code ---

    def test_chunk_basic_no_overlap(self):
        """Test basic chunking with no overlap."""
        code = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6"
        expected_chunks = ["Line 1\nLine 2\nLine 3", "Line 4\nLine 5\nLine 6"]
        self.assertEqual(chunk_code(code, 3, 0), expected_chunks)

        code_2 = "L1\nL2\nL3\nL4\nL5"
        expected_chunks_2 = ["L1\nL2", "L3\nL4", "L5"]
        self.assertEqual(chunk_code(code_2, 2, 0), expected_chunks_2)


    def test_chunk_with_overlap(self):
        """Test chunking with overlap."""
        code = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6\nLine 7"
        # Chunk 1: L1, L2, L3, L4
        # Chunk 2: L3, L4, L5, L6 (starts 4-2=2 lines from end of C1, so L3)
        # Chunk 3: L5, L6, L7 (starts 4-2=2 lines from end of C2, so L5)
        expected_chunks = [
            "Line 1\nLine 2\nLine 3\nLine 4",
            "Line 3\nLine 4\nLine 5\nLine 6",
            "Line 5\nLine 6\nLine 7"
        ]
        self.assertEqual(chunk_code(code, 4, 2), expected_chunks)

        code_short_overlap = "L1\nL2\nL3\nL4\nL5"
        # C1: L1,L2,L3
        # C2: L2,L3,L4
        # C3: L3,L4,L5
        # C4: L4,L5
        # C5: L5
        # The logic in file_utils.py for chunk_code for `next_start_pos <= current_pos` was:
        # `next_start_pos = current_pos + 1`
        # This might affect behavior with very high overlap or small chunks.
        # Let's re-verify expected based on current implementation:
        # current_pos=0, end_pos=3 -> C1: L1,L2,L3. next_start_pos = 3-2=1.
        # current_pos=1, end_pos=4 -> C2: L2,L3,L4. next_start_pos = 4-2=2.
        # current_pos=2, end_pos=5 -> C3: L3,L4,L5. next_start_pos = 5-2=3.
        # current_pos=3, end_pos=6 -> C4: L4,L5. next_start_pos = 6-2=4. (end_pos > len(lines), chunk is lines[3:5])
        # current_pos=4, end_pos=7 -> C5: L5. (end_pos > len(lines), chunk is lines[4:5])
        # Loop breaks because end_pos >= len(lines) in the iteration that produced C5.
        expected_short_overlap = [
            "L1\nL2\nL3", # pos=0, end=3. next_start=1
            "L2\nL3\nL4", # pos=1, end=4. next_start=2
            "L3\nL4\nL5", # pos=2, end=5. next_start=3. Loop ends after this chunk as end_pos >= len(lines)
        ]
        # After re-checking the `chunk_code` logic:
        # The loop continues as long as `current_pos < len(lines)`.
        # `end_pos` is `current_pos + lines_per_chunk`.
        # `next_start_pos` is `end_pos - overlap_lines`.
        # If `end_pos >= len(lines)`, it breaks *after* processing the current chunk.

        # For code_short_overlap ("L1\nL2\nL3\nL4\nL5", 3, 2):
        # lines = ["L1", "L2", "L3", "L4", "L5"] (len=5)
        # 1. current_pos = 0. end_pos = 0 + 3 = 3. chunk_lines = lines[0:3] = ["L1", "L2", "L3"]. chunks.append("L1\nL2\nL3").
        #    next_start_pos = 3 - 2 = 1. end_pos (3) < len(lines) (5). current_pos = 1.
        # 2. current_pos = 1. end_pos = 1 + 3 = 4. chunk_lines = lines[1:4] = ["L2", "L3", "L4"]. chunks.append("L2\nL3\nL4").
        #    next_start_pos = 4 - 2 = 2. end_pos (4) < len(lines) (5). current_pos = 2.
        # 3. current_pos = 2. end_pos = 2 + 3 = 5. chunk_lines = lines[2:5] = ["L3", "L4", "L5"]. chunks.append("L3\nL4\nL5").
        #    next_start_pos = 5 - 2 = 3. end_pos (5) >= len(lines) (5). Break.
        # So, expected_short_overlap is correct.
        self.assertEqual(chunk_code(code_short_overlap, 3, 2), expected_short_overlap)


    def test_chunk_empty_string(self):
        """Test chunking an empty string."""
        self.assertEqual(chunk_code("", 10, 2), [])

    def test_chunk_content_shorter_than_chunk_size(self):
        """Test chunking content shorter than lines_per_chunk."""
        code = "Line 1\nLine 2"
        expected_chunks = ["Line 1\nLine 2"]
        self.assertEqual(chunk_code(code, 5, 1), expected_chunks)
        
        code_single_line = "Single line only"
        expected_single = ["Single line only"]
        self.assertEqual(chunk_code(code_single_line, 5, 1), expected_single)

    def test_chunk_invalid_lines_per_chunk(self):
        """Test input validation for lines_per_chunk."""
        with self.assertRaisesRegex(ValueError, "lines_per_chunk must be a positive integer."):
            chunk_code("some code", 0, 2)
        with self.assertRaisesRegex(ValueError, "lines_per_chunk must be a positive integer."):
            chunk_code("some code", -1, 2)
        with self.assertRaisesRegex(ValueError, "lines_per_chunk must be a positive integer."):
            chunk_code("some code", 0.5, 2) # type check

    def test_chunk_invalid_overlap_lines(self):
        """Test input validation for overlap_lines."""
        with self.assertRaisesRegex(ValueError, "overlap_lines must be a non-negative integer."):
            chunk_code("some code", 5, -1)
        with self.assertRaisesRegex(ValueError, "overlap_lines must be a non-negative integer."):
            chunk_code("some code", 5, -0.5) # type check

        with self.assertRaisesRegex(ValueError, "overlap_lines must be less than lines_per_chunk."):
            chunk_code("some code", 5, 5) # overlap == lines_per_chunk
        with self.assertRaisesRegex(ValueError, "overlap_lines must be less than lines_per_chunk."):
            chunk_code("some code", 5, 6) # overlap > lines_per_chunk

if __name__ == '__main__':
    # Fallback sys.path modification if needed, similar to test_directory_scanner
    if not any("file_utils" in m for m in sys.modules):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))) # Check current dir
        try:
            from file_utils import read_file_content, chunk_code
        except ImportError:
             # If file_utils.py is in the root of the repo, and tests are in a subdir
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            if project_root not in sys.path:
                 sys.path.insert(0, project_root)
            from file_utils import read_file_content, chunk_code

    unittest.main(verbosity=2)
