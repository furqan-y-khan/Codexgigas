import os

def read_file_content(filepath: str) -> str:
    """
    Reads the entire content of the file specified by filepath.

    Args:
        filepath: The path to the file.

    Returns:
        The content of the file as a string, or an error message string
        if an error occurs.
    """
    if not os.path.exists(filepath):
        return f"Error: File not found at {filepath}"
    
    if os.path.isdir(filepath):
        return f"Error: Path is a directory, not a file at {filepath}"
    
    if not os.path.isfile(filepath): # Should be redundant due to above checks, but good for explicit clarity
        return f"Error: Path is not a file at {filepath}"

    try:
        with open(filepath, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError: # This case should ideally be caught by os.path.exists, but good for robustness
        return f"Error: File not found at {filepath}"
    except IOError:
        return f"Error: Could not read file at {filepath} due to an IO error."
    except Exception as e: # Catch any other unexpected errors
        return f"Error: An unexpected error occurred while reading {filepath}: {e}"

def chunk_code(code_string: str, lines_per_chunk: int, overlap_lines: int) -> list[str]:
    """
    Chunks a string of code into overlapping segments.

    Args:
        code_string: The string of code to chunk (can be multi-line).
        lines_per_chunk: The number of lines each chunk should ideally have.
        overlap_lines: The number of lines to overlap between consecutive chunks.

    Returns:
        A list of strings, where each string is a chunk of code.
        Returns an empty list if code_string is empty.
        Returns the original code_string as a single chunk if it has fewer
        lines than lines_per_chunk.

    Raises:
        ValueError: If lines_per_chunk <= 0, overlap_lines < 0, or
                    overlap_lines >= lines_per_chunk.
    """
    if not isinstance(lines_per_chunk, int) or lines_per_chunk <= 0:
        raise ValueError("lines_per_chunk must be a positive integer.")
    if not isinstance(overlap_lines, int) or overlap_lines < 0:
        raise ValueError("overlap_lines must be a non-negative integer.")
    if overlap_lines >= lines_per_chunk:
        raise ValueError("overlap_lines must be less than lines_per_chunk.")

    lines = code_string.splitlines()
    if not lines:
        return []

    if len(lines) <= lines_per_chunk:
        return ["\n".join(lines)]

    chunks = []
    current_pos = 0
    while current_pos < len(lines):
        end_pos = current_pos + lines_per_chunk
        chunk_lines = lines[current_pos:end_pos]
        chunks.append("\n".join(chunk_lines))
        
        next_start_pos = end_pos - overlap_lines
        
        # If the next chunk would be identical to the current one (due to overlap)
        # or if we are at the end and the next chunk would be empty or very small
        if next_start_pos <= current_pos and end_pos < len(lines) : 
             # This can happen if lines_per_chunk is small and overlap_lines is large (relative to lines_per_chunk)
             # Or if we are stuck. To prevent infinite loop, advance by at least one line if not advancing.
             next_start_pos = current_pos + 1 


        if end_pos >= len(lines): # Reached or passed the end of lines
            break 
        
        current_pos = next_start_pos
        if current_pos >= len(lines) and chunks[-1] != "\n".join(lines[len(lines)-lines_per_chunk if len(lines)-lines_per_chunk > 0 else 0:]):
            # This ensures the last part of the code is captured if the loop terminates early due to overlap logic
            # and the last chunk isn't already the full tail end.
            # However, we need to be careful not to add a duplicate of the last chunk if it already covers the end.
            # The condition current_pos >= len(lines) means we are about to exit
            # The second condition checks if the very last generated chunk contains the last lines of the input
            # If not, we might need to add one more chunk.
            # A simpler way is to check if the last line of the input is in the last chunk.
            # If not lines[-1] in chunks[-1].splitlines()[-1] : # This is too simplistic
            pass # The current logic should handle this correctly by breaking when end_pos >= len(lines)

    return chunks


if __name__ == "__main__":
    # --- Tests for read_file_content ---
    print("--- Testing read_file_content ---")
    dummy_file_name = "test_file.txt"
    dummy_dir_name = "test_dir_for_utils"

    print(f"1. Creating dummy file: {dummy_file_name}")
    with open(dummy_file_name, "w") as f:
        f.write("Hello, world!\nThis is a test file.")
    
    print(f"2. Creating dummy directory: {dummy_dir_name}")
    if not os.path.exists(dummy_dir_name):
        os.makedirs(dummy_dir_name)

    print(f"\n3. Testing successful read of {dummy_file_name}:")
    content = read_file_content(dummy_file_name)
    print(f"Content:\n---\n{content}\n---")

    non_existent_file = "non_existent_file.txt"
    print(f"\n4. Testing non-existent file: {non_existent_file}")
    error_message = read_file_content(non_existent_file)
    print(error_message)

    print(f"\n5. Testing with a directory path: {dummy_dir_name}")
    error_message_dir = read_file_content(dummy_dir_name)
    print(error_message_dir)
    
    print(f"\n6. Cleaning up for read_file_content tests...")
    if os.path.exists(dummy_file_name):
        os.remove(dummy_file_name)
    if os.path.exists(dummy_dir_name):
        os.rmdir(dummy_dir_name)
    print("Cleanup for read_file_content complete.")

    # --- Tests for chunk_code ---
    print("\n\n--- Testing chunk_code ---")
    sample_code = "\n".join([f"Line {i+1}" for i in range(20)]) # 20 lines of code

    print("\n1. Standard chunking (10 lines per chunk, 2 overlap):")
    chunks = chunk_code(sample_code, 10, 2)
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}:\n---\n{chunk}\n---")
    
    # Expected:
    # Chunk 1: Lines 1-10
    # Chunk 2: Lines 9-18
    # Chunk 3: Lines 17-20 (or lines 17 - end, if total lines < 18+10-2 = 26)
    # Actual test for chunk 3:
    # Last chunk should contain "Line 20"
    if chunks and "Line 20" not in chunks[-1]:
        print("Error: Last line ('Line 20') not found in the last chunk for standard test.")
    elif not chunks:
        print("Error: No chunks returned for standard test.")


    print("\n2. Fewer lines than lines_per_chunk (5 lines input, 10 per chunk, 2 overlap):")
    short_code = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
    chunks_short = chunk_code(short_code, 10, 2)
    for i, chunk in enumerate(chunks_short):
        print(f"  Chunk {i+1}:\n---\n{chunk}\n---")
    if len(chunks_short) == 1 and chunks_short[0] == short_code:
        print("  Correctly returned the whole string as one chunk.")
    else:
        print(f"  Incorrect output for short code. Expected 1 chunk, got {len(chunks_short)}")

    print("\n3. Empty input string:")
    chunks_empty = chunk_code("", 10, 2)
    print(f"  Chunks for empty string: {chunks_empty}")
    if chunks_empty == []:
        print("  Correctly returned an empty list.")
    else:
        print("  Incorrect output for empty string.")

    print("\n4. Test lines_per_chunk = total lines (20 lines input, 20 per chunk, 2 overlap):")
    chunks_exact = chunk_code(sample_code, 20, 2)
    for i, chunk in enumerate(chunks_exact):
        print(f"  Chunk {i+1}:\n---\n{chunk}\n---")
    if len(chunks_exact) == 1 and chunks_exact[0] == sample_code:
        print("  Correctly returned the whole string as one chunk.")
    else:
        print(f"  Incorrect output for exact lines. Expected 1 chunk, got {len(chunks_exact)}")


    print("\n5. Test lines_per_chunk slightly less than total (18 lines per chunk, 2 overlap for 20 lines input):")
    chunks_less = chunk_code(sample_code, 18, 2)
    for i, chunk in enumerate(chunks_less):
        print(f"  Chunk {i+1}:\n---\n{chunk}\n---")
    # Expected: Chunk 1 (1-18), Chunk 2 (17-20)
    if len(chunks_less) == 2 and "Line 18" in chunks_less[0] and "Line 17" in chunks_less[1] and "Line 20" in chunks_less[1]:
        print("  Correctly chunked for slightly less lines per chunk.")
    else:
        print("  Incorrect output for slightly less lines per chunk.")


    print("\n6. Test zero overlap (10 lines per chunk, 0 overlap):")
    chunks_no_overlap = chunk_code(sample_code, 10, 0)
    for i, chunk in enumerate(chunks_no_overlap):
        print(f"  Chunk {i+1}:\n---\n{chunk}\n---")
    # Expected: Chunk 1 (1-10), Chunk 2 (11-20)
    if len(chunks_no_overlap) == 2 and "Line 10" in chunks_no_overlap[0] and "Line 11" in chunks_no_overlap[1] and "Line 20" in chunks_no_overlap[1]:
         print("  Correctly chunked with zero overlap.")
    else:
        print("  Incorrect output for zero overlap.")


    print("\n7. Test large overlap (5 lines per chunk, 4 overlap for 10 lines):")
    ten_lines = "\n".join([f"Line {i+1}" for i in range(10)])
    chunks_large_overlap = chunk_code(ten_lines, 5, 4)
    # Expected:
    # C1: 1-5
    # C2: 2-6
    # C3: 3-7
    # C4: 4-8
    # C5: 5-9
    # C6: 6-10
    for i, chunk in enumerate(chunks_large_overlap):
        print(f"  Chunk {i+1} (LPC=5, OL=4):\n---\n{chunk}\n---")
    if chunks_large_overlap and "Line 10" in chunks_large_overlap[-1] and len(chunks_large_overlap) == 6:
        print(" Correctly chunked with large overlap.")
    else:
        print(f" Incorrect output for large overlap. Got {len(chunks_large_overlap)} chunks.")


    print("\n8. Validation Tests:")
    try:
        print("  Testing lines_per_chunk = 0:")
        chunk_code(sample_code, 0, 2)
    except ValueError as e:
        print(f"  Caught expected error: {e}")

    try:
        print("  Testing overlap_lines < 0:")
        chunk_code(sample_code, 10, -1)
    except ValueError as e:
        print(f"  Caught expected error: {e}")

    try:
        print("  Testing overlap_lines >= lines_per_chunk (overlap == lpc):")
        chunk_code(sample_code, 5, 5)
    except ValueError as e:
        print(f"  Caught expected error: {e}")
    
    try:
        print("  Testing overlap_lines >= lines_per_chunk (overlap > lpc):")
        chunk_code(sample_code, 5, 6)
    except ValueError as e:
        print(f"  Caught expected error: {e}")

    print("\nAll chunk_code tests complete.")
