import os
import shutil
import sys
import random # Added for generate_simulated_embedding
import json   # Added for saving to JSON

# Adjust sys.path to find sibling modules directory_scanner and file_utils
# This assumes code_processor.py, directory_scanner.py, and file_utils.py are in the same directory
# or the project root. For more complex structures, PYTHONPATH or packaging would be better.
current_script_dir = os.path.dirname(os.path.abspath(__file__))
if current_script_dir not in sys.path:
    sys.path.append(current_script_dir)

# Attempt to import from sibling modules
try:
    from directory_scanner import scan_directory
    from file_utils import read_file_content, chunk_code
except ImportError as e:
    print(f"Error importing modules. Ensure directory_scanner.py and file_utils.py are accessible.")
    print(f"Details: {e}")
    # As a fallback, if they are in the parent directory (e.g. if this script is in a 'tools' subdir)
    parent_dir = os.path.dirname(current_script_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    try:
        from directory_scanner import scan_directory
        from file_utils import read_file_content, chunk_code
    except ImportError as e_inner:
        print(f"Error importing modules from parent directory either. Please check structure.")
        print(f"Details: {e_inner}")
        raise

def generate_simulated_embedding(text: str, embedding_dim: int = 10) -> list[float]:
    """
    Generates a simulated language model embedding for the given text.

    Args:
        text: The input string (e.g., a code chunk).
        embedding_dim: The desired dimensionality of the embedding. Defaults to 10.

    Returns:
        A list of `embedding_dim` random floating-point numbers between 0.0 and 1.0.
    """
    if not isinstance(embedding_dim, int) or embedding_dim <= 0:
        raise ValueError("embedding_dim must be a positive integer.")
    return [random.random() for _ in range(embedding_dim)]

def process_project_files(
    project_path: str, 
    file_extensions: list[str], 
    lines_per_chunk: int, 
    overlap_lines: int,
    embedding_dim: int = 10
) -> list[dict]:
    """
    Scans a project directory, filters for specified file types, reads their content,
    chunks the content, generates simulated embeddings, and returns a list of dictionaries.
    (Details as before)
    """
    scanned_items = scan_directory(project_path)
    
    processed_chunks = []
    normalized_extensions = [ext.lower() for ext in file_extensions]

    for item in scanned_items:
        if item['type'] == 'file':
            filename, file_ext = os.path.splitext(item['path'])
            if file_ext.lower() in normalized_extensions:
                content_or_error = read_file_content(item['path'])

                if content_or_error.startswith("Error:"):
                    print(f"Warning: Skipping file {item['path']} due to read error: {content_or_error}")
                    continue
                
                try:
                    chunks = chunk_code(content_or_error, lines_per_chunk, overlap_lines)
                    for chunk_text in chunks:
                        embedding = generate_simulated_embedding(chunk_text, embedding_dim)
                        processed_chunks.append({
                            'filepath': item['path'], 
                            'chunk_text': chunk_text,
                            'embedding': embedding
                        })
                except ValueError as e: 
                    print(f"Warning: Skipping file {item['path']} due to processing error: {e}")
                    continue
                    
    return processed_chunks

def save_processed_data_to_json(processed_data: list[dict], output_filepath: str) -> None:
    """
    Saves the processed data (list of dictionaries) to a JSON file.

    Args:
        processed_data: The data to save, typically a list of dictionaries.
        output_filepath: The path to the file where the JSON data will be saved.
    """
    try:
        # Ensure the directory for the output file exists
        output_dir = os.path.dirname(output_filepath)
        if output_dir: # Check if output_dir is not empty (i.e., not saving in current dir directly)
            os.makedirs(output_dir, exist_ok=True)
            
        with open(output_filepath, 'w') as f:
            json.dump(processed_data, f, indent=4)
        print(f"Successfully saved processed data to {output_filepath}")
    except IOError as e:
        print(f"Error: Could not write JSON to {output_filepath}. IO Error: {e}")
    except TypeError as e:
        print(f"Error: Could not serialize data to JSON for {output_filepath}. Type Error: {e}")
    except Exception as e: # Catch any other unexpected errors during saving
        print(f"Error: An unexpected error occurred while saving JSON to {output_filepath}: {e}")


if __name__ == "__main__":
    temp_project_dir = os.path.abspath("temp_dummy_project_embeddings_json") # New name

    module1_dir = os.path.join(temp_project_dir, "module1")
    module2_dir = os.path.join(temp_project_dir, "module2")
    if os.path.exists(temp_project_dir): 
        shutil.rmtree(temp_project_dir)
    os.makedirs(module1_dir, exist_ok=True)
    os.makedirs(module2_dir, exist_ok=True)

    py_content_1 = "\n".join([f"def func1_line_{i+1}(): # Python" for i in range(15)]) + \
                   "\n" + "print('Hello from func1')"
    py_content_2 = "\n".join([f"# Module 2, line {i+1} # Python" for i in range(8)]) + \
                   "\nclass MyClass:\n    pass\n# End of class" 
    py_content_3 = "import os\n# Short script # Python\nprint(os.getcwd())"

    with open(os.path.join(module1_dir, "file1.py"), "w") as f: f.write(py_content_1)
    with open(os.path.join(module1_dir, "notes.txt"), "w") as f: f.write("This is a text file.\nIt has a few lines.")
    with open(os.path.join(module2_dir, "file2.py"), "w") as f: f.write(py_content_2)
    with open(os.path.join(temp_project_dir, "main_script.py"), "w") as f: f.write(py_content_3)
    with open(os.path.join(temp_project_dir, "README.md"), "w") as f: f.write("# Project Readme\n## Markdown file")
    
    unreadable_file_path = os.path.join(module1_dir, "unreadable.py")
    with open(unreadable_file_path, "w") as f: f.write("# I am unreadable")
    try:
        os.chmod(unreadable_file_path, 0o000)
    except Exception as e:
        print(f"Warning: Could not set {unreadable_file_path} to unreadable: {e}")

    print(f"Created dummy project at: {temp_project_dir}")
    test_embedding_dim = 3 # Reduced for simpler JSON output in example

    try:
        print(f"\n--- Processing .py files (10 lines/chunk, 2 overlap, {test_embedding_dim}-dim embeddings) ---")
        py_processed_data = process_project_files(
            project_path=temp_project_dir,
            file_extensions=['.py'],
            lines_per_chunk=10,
            overlap_lines=2,
            embedding_dim=test_embedding_dim
        )

        if py_processed_data:
            for item in py_processed_data: # Keep printing for console verification
                print(f"\nFile: {item['filepath']}")
                print("Chunk:")
                print("```")
                print(item['chunk_text'])
                print("```")
                embedding_display_str = f"[{', '.join(map(lambda x: f'{x:.4f}', item['embedding']))}]"
                print(f"Simulated Embedding ({len(item['embedding'])} dims): {embedding_display_str}")
            
            # Save the .py processed data to JSON
            output_json_filepath = os.path.join(temp_project_dir, "processed_python_code.json")
            print(f"\nAttempting to save Python processed data to: {output_json_filepath}")
            save_processed_data_to_json(py_processed_data, output_json_filepath)

        else:
            print("No .py file data processed.")

        # (Keep other tests as they were, for brevity I'm not repeating them here, but they'd be in the full script)
        print(f"\n--- Processing .txt files (5 lines/chunk, 1 overlap, {test_embedding_dim}-dim embeddings) ---")
        # ... (txt_processed_data and its printing logic)
        txt_processed_data = process_project_files(
            project_path=temp_project_dir,
            file_extensions=['.txt'],
            lines_per_chunk=5,
            overlap_lines=1,
            embedding_dim=test_embedding_dim
        )
        if txt_processed_data:
            # (Optional: print txt_processed_data to console)
            output_txt_json_filepath = os.path.join(temp_project_dir, "processed_text_files.json")
            print(f"\nAttempting to save TXT processed data to: {output_txt_json_filepath}")
            save_processed_data_to_json(txt_processed_data, output_txt_json_filepath)
        else:
            print("No .txt file data processed.")

        print("\n--- Testing with invalid project path ---")
        try:
            process_project_files("non_existent_project_path", [".py"], 10, 2, embedding_dim=test_embedding_dim)
        except FileNotFoundError as e:
            print(f"Correctly caught error: {e}")
        
        # ... (other tests like invalid chunk params, invalid embedding dim)

    finally:
        if os.path.exists(unreadable_file_path):
            try:
                os.chmod(unreadable_file_path, 0o777)
            except Exception as e:
                print(f"Warning: Could not change permissions of {unreadable_file_path} for cleanup: {e}")
        
        if os.path.exists(temp_project_dir):
            print(f"\nCleaning up dummy project: {temp_project_dir}")
            shutil.rmtree(temp_project_dir)
            print("Cleanup complete.")
