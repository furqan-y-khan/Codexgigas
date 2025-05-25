import os
import shutil
import sys
import random # Added for generate_simulated_embedding
import json   # Added for saving to JSON
import numpy as np # Added for FAISS integration

# Adjust sys.path to find sibling modules
current_script_dir = os.path.dirname(os.path.abspath(__file__))
if current_script_dir not in sys.path:
    sys.path.append(current_script_dir)

# Attempt to import from sibling modules
try:
    from directory_scanner import scan_directory
    from file_utils import read_file_content, chunk_code
    # Import FAISS utility functions and availability flag
    from vector_db_utils import (
        initialize_faiss_index, 
        add_embeddings_to_index,
        save_faiss_index,
        search_faiss_index,
        FAISS_AVAILABLE # This flag is crucial
    )
except ImportError as e:
    print(f"Error importing modules. Ensure directory_scanner.py, file_utils.py, and vector_db_utils.py are accessible.")
    print(f"Details: {e}")
    # Fallback for parent directory structure (e.g., if this script is in a 'tools' subdir)
    parent_dir = os.path.dirname(current_script_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    try:
        from directory_scanner import scan_directory
        from file_utils import read_file_content, chunk_code
        from vector_db_utils import (
            initialize_faiss_index, add_embeddings_to_index,
            save_faiss_index, search_faiss_index, FAISS_AVAILABLE
        )
    except ImportError as e_inner:
        print(f"Error importing modules from parent directory either. Please check structure.")
        print(f"Details: {e_inner}")
        # Set FAISS_AVAILABLE to False if vector_db_utils itself can't be imported,
        # so the rest of the script can gracefully handle FAISS absence.
        FAISS_AVAILABLE = False 
        print("Warning: vector_db_utils not found. FAISS functionalities will be disabled.")
        # We might still want to run without FAISS, so don't raise, but other imports might be critical.
        # For this specific task, let's assume directory_scanner and file_utils must exist.
        if "directory_scanner" not in str(e_inner) and "file_utils" not in str(e_inner):
             raise e_inner


def generate_simulated_embedding(text: str, embedding_dim: int = 10) -> list[float]:
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
    scanned_items = scan_directory(project_path)
    processed_data = [] # Renamed from processed_chunks for clarity
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
                        processed_data.append({
                            'filepath': item['path'], 
                            'chunk_text': chunk_text,
                            'embedding': embedding
                        })
                except ValueError as e: 
                    print(f"Warning: Skipping file {item['path']} due to processing error: {e}")
                    continue
    
    # --- FAISS Integration ---
    if FAISS_AVAILABLE and processed_data:
        all_embeddings = [data_item['embedding'] for data_item in processed_data if data_item.get('embedding')]
        
        if not all_embeddings:
            print("No embeddings found in processed data. Skipping FAISS indexing.")
        else:
            try:
                embedding_matrix = np.array(all_embeddings, dtype=np.float32)
                if embedding_matrix.ndim == 1: # Handle case of single embedding
                    embedding_matrix = embedding_matrix.reshape(1, -1)

                if embedding_matrix.size == 0: # Should be caught by `if not all_embeddings` but as safety
                    print("Embedding matrix is empty. Skipping FAISS indexing.")
                else:
                    current_embedding_dim = embedding_matrix.shape[1]
                    
                    print(f"\n--- FAISS Indexing (Dimension: {current_embedding_dim}) ---")
                    faiss_index = initialize_faiss_index(current_embedding_dim)

                    if faiss_index:
                        add_embeddings_to_index(faiss_index, embedding_matrix)
                        
                        # Define index output path within the project directory for cleanup
                        project_name = os.path.basename(os.path.abspath(project_path))
                        index_output_filename = f"{project_name}_embeddings.faiss"
                        index_output_filepath = os.path.join(project_path, index_output_filename) # Save inside temp project
                        
                        save_faiss_index(faiss_index, index_output_filepath)
                        print(f"FAISS index saved at: {index_output_filepath}")

                        # Optional Search Demonstration
                        if embedding_matrix.shape[0] > 0:
                            print("\n--- FAISS Search Demonstration ---")
                            query_embedding = embedding_matrix[0] # Use the first embedding as query
                            top_k = min(3, embedding_matrix.shape[0]) # Search for top_k or num_embeddings if fewer
                            
                            search_results = search_faiss_index(faiss_index, query_embedding, top_k)
                            if search_results:
                                D, I = search_results
                                print(f"Search results for the first chunk (embedding 0), top {top_k}:")
                                for i in range(I.shape[1]):
                                    result_idx = I[0][i]
                                    distance = D[0][i]
                                    original_data_item = processed_data[result_idx]
                                    print(f"  Neighbor {i+1}:")
                                    print(f"    Index: {result_idx}, Distance: {distance:.4f}")
                                    print(f"    Filepath: {original_data_item['filepath']}")
                                    print(f"    Chunk Text Snippet: '{original_data_item['chunk_text'][:100].replace('\n', ' ')}...'")
                            else:
                                print("Search returned no results or failed.")
                    else:
                        print("FAISS index initialization failed. Skipping FAISS operations.")
            except Exception as e: # Catch any unexpected error during FAISS processing
                print(f"An error occurred during FAISS processing: {e}")
    elif not FAISS_AVAILABLE:
        print("\nFAISS library not available. Skipping FAISS indexing and search.")
    elif not processed_data:
        print("\nNo data processed. Skipping FAISS indexing and search.")
                    
    return processed_data

def save_processed_data_to_json(processed_data: list[dict], output_filepath: str) -> None:
    try:
        output_dir = os.path.dirname(output_filepath)
        if output_dir: 
            os.makedirs(output_dir, exist_ok=True)
        with open(output_filepath, 'w') as f:
            json.dump(processed_data, f, indent=4)
        print(f"Successfully saved processed data to {output_filepath}")
    except IOError as e:
        print(f"Error: Could not write JSON to {output_filepath}. IO Error: {e}")
    except TypeError as e:
        print(f"Error: Could not serialize data to JSON for {output_filepath}. Type Error: {e}")
    except Exception as e: 
        print(f"Error: An unexpected error occurred while saving JSON to {output_filepath}: {e}")


if __name__ == "__main__":
    # temp_project_dir should be unique for this main block if running multiple times
    temp_project_dir = os.path.abspath("temp_dummy_project_faiss_integration") 

    module1_dir = os.path.join(temp_project_dir, "module1")
    module2_dir = os.path.join(temp_project_dir, "module2")
    if os.path.exists(temp_project_dir): 
        shutil.rmtree(temp_project_dir)
    os.makedirs(module1_dir, exist_ok=True)
    os.makedirs(module2_dir, exist_ok=True)

    # Create sample files with enough lines for chunking and diverse content for search
    py_content_1 = "def main_function():\n" + "\n".join([f"    # Python code line {i+1}" for i in range(20)]) + \
                   "\n    print('Primary execution point')" # 22 lines
    py_content_2 = "class HelperClass:\n" + "\n".join([f"    # Utility method line {i+1}" for i in range(15)]) + \
                   "\n    def assist():\n        return True" # 18 lines
    py_content_3 = "import sys\n# A very short script\n# Used for simple tasks\nprint(sys.version)" # 4 lines

    with open(os.path.join(module1_dir, "main_app.py"), "w") as f: f.write(py_content_1)
    with open(os.path.join(module1_dir, "notes_module1.txt"), "w") as f: f.write("Text file with notes about module 1.\n" * 5)
    with open(os.path.join(module2_dir, "utils.py"), "w") as f: f.write(py_content_2)
    with open(os.path.join(temp_project_dir, "short_script.py"), "w") as f: f.write(py_content_3)
    with open(os.path.join(temp_project_dir, "README.md"), "w") as f: f.write("# Project Readme\n## Markdown file for docs.")
    
    unreadable_file_path = os.path.join(module1_dir, "unreadable.py")
    with open(unreadable_file_path, "w") as f: f.write("# I am unreadable")
    try:
        os.chmod(unreadable_file_path, 0o000)
    except Exception as e:
        print(f"Warning: Could not set {unreadable_file_path} to unreadable: {e}")

    print(f"Created dummy project at: {temp_project_dir}")
    test_embedding_dim = 5 
    random.seed(42) # For predictable embeddings if FAISS is available

    try:
        print(f"\n--- Processing .py files (10 lines/chunk, 2 overlap, {test_embedding_dim}-dim embeddings) ---")
        py_processed_data = process_project_files(
            project_path=temp_project_dir,
            file_extensions=['.py'],
            lines_per_chunk=10, # Ensure some files get multiple chunks
            overlap_lines=2,
            embedding_dim=test_embedding_dim
        )

        if py_processed_data:
            # The function process_project_files will now print FAISS related info internally
            # We can still print a summary of what was processed here if desired.
            print(f"\nTotal processed items (chunks) for .py files: {len(py_processed_data)}")
            if len(py_processed_data) > 0:
                 print(f"Example processed item: File='{py_processed_data[0]['filepath']}', Chunk snippet='{py_processed_data[0]['chunk_text'][:30].replace('\n',' ')}...'")
            
            output_json_filepath = os.path.join(temp_project_dir, "processed_python_code.json")
            print(f"\nAttempting to save Python processed data to: {output_json_filepath}")
            save_processed_data_to_json(py_processed_data, output_json_filepath)
        else:
            print("No .py file data processed or FAISS processing skipped some steps.")

        # (Other tests like .txt files, invalid paths, etc., can be added or kept as before)

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
