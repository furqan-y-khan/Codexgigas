import numpy as np
import os # For cleanup in example

# Attempt to import faiss and set a global flag
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    # This warning will be printed once when the module is first imported if FAISS is not found.
    print("Warning: FAISS library not found. Vector database functionalities will be disabled.")

def initialize_faiss_index(embedding_dimension: int) -> 'faiss.Index | None':
    """
    Initializes a FAISS index.

    Args:
        embedding_dimension: The dimension of the embeddings to be stored.

    Returns:
        A FAISS index object or None if FAISS is not available.
    """
    if not FAISS_AVAILABLE:
        print("Warning: FAISS library not available, cannot initialize index.")
        return None
    try:
        return faiss.IndexFlatL2(embedding_dimension)
    except Exception as e:
        print(f"Error initializing FAISS index: {e}")
        return None

def add_embeddings_to_index(index: 'faiss.Index | None', embeddings: np.ndarray) -> None:
    """
    Adds embeddings to the FAISS index.

    Args:
        index: The FAISS index object.
        embeddings: A 2D NumPy array of embeddings (each row is an embedding).
    """
    if not FAISS_AVAILABLE or index is None:
        print("Warning: FAISS index not available or not initialized, cannot add embeddings.")
        return
    
    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        print("Error: Embeddings must be a 2D NumPy array.")
        return
        
    try:
        embeddings_float32 = embeddings.astype(np.float32)
        index.add(embeddings_float32)
    except Exception as e:
        print(f"Error adding embeddings to FAISS index: {e}")

def search_faiss_index(index: 'faiss.Index | None', query_embedding: np.ndarray, top_k: int) -> 'tuple[np.ndarray, np.ndarray] | None':
    """
    Searches the FAISS index for the nearest neighbors to a query embedding.

    Args:
        index: The FAISS index object.
        query_embedding: A 1D NumPy array representing the query embedding.
        top_k: The number of nearest neighbors to find.

    Returns:
        A tuple (D, I) where D is an array of distances and I is an array of indices
        of the nearest neighbors, or None if FAISS is not available or search fails.
    """
    if not FAISS_AVAILABLE or index is None:
        print("Warning: FAISS index not available or not initialized, cannot perform search.")
        return None

    if not isinstance(query_embedding, np.ndarray) or query_embedding.ndim != 1:
        print("Error: Query embedding must be a 1D NumPy array.")
        return None
    
    try:
        query_embedding_float32 = query_embedding.astype(np.float32).reshape(1, -1)
        if query_embedding_float32.shape[1] != index.d:
            print(f"Error: Query embedding dimension ({query_embedding_float32.shape[1]}) "
                  f"does not match index dimension ({index.d}).")
            return None
        D, I = index.search(query_embedding_float32, top_k)
        return D, I
    except Exception as e:
        print(f"Error searching FAISS index: {e}")
        return None

def save_faiss_index(index: 'faiss.Index | None', index_filepath: str) -> None:
    """
    Saves the FAISS index to a file.

    Args:
        index: The FAISS index object.
        index_filepath: The path to the file where the index will be saved.
    """
    if not FAISS_AVAILABLE or index is None:
        print("Warning: FAISS index not available or not initialized, cannot save index.")
        return
    try:
        # Ensure directory exists
        output_dir = os.path.dirname(index_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        faiss.write_index(index, index_filepath)
        print(f"FAISS index saved to {index_filepath}")
    except IOError as e:
        print(f"Error: Could not write FAISS index to {index_filepath}. IO Error: {e}")
    except Exception as e: # faiss.write_index can raise other faiss-specific exceptions
        print(f"Error saving FAISS index to {index_filepath}: {e}")


def load_faiss_index(index_filepath: str) -> 'faiss.Index | None':
    """
    Loads a FAISS index from a file.

    Args:
        index_filepath: The path to the file from which to load the index.

    Returns:
        A FAISS index object or None if FAISS is not available or loading fails.
    """
    if not FAISS_AVAILABLE:
        print("Warning: FAISS library not available, cannot load index.")
        return None
    
    if not os.path.exists(index_filepath):
        print(f"Error: Index file not found at {index_filepath}")
        return None
        
    try:
        index = faiss.read_index(index_filepath)
        print(f"FAISS index loaded from {index_filepath}")
        return index
    except (RuntimeError, IOError) as e: # RuntimeError for bad index file, IOError for file access
        print(f"Error loading FAISS index from {index_filepath}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred loading FAISS index from {index_filepath}: {e}")
        return None

if __name__ == "__main__":
    print("\n--- FAISS Vector DB Utils Demonstration ---")

    embedding_dim = 10
    num_embeddings = 5
    dummy_index_file = "dummy_faiss.index"

    # 1. Initialize index
    print("\n1. Initializing FAISS index...")
    index = initialize_faiss_index(embedding_dim)

    if index is not None:
        print(f"FAISS index initialized. Dimension: {index.d}, Is_trained: {index.is_trained}, Total: {index.ntotal}")

        # 2. Create dummy embeddings and add them
        print("\n2. Creating and adding dummy embeddings...")
        dummy_embeddings = np.random.rand(num_embeddings, embedding_dim).astype(np.float32)
        print(f"Shape of dummy embeddings: {dummy_embeddings.shape}")
        add_embeddings_to_index(index, dummy_embeddings)
        print(f"Embeddings added to index. Index total: {index.ntotal}")

        # 3. Perform a search
        print("\n3. Performing a search...")
        if index.ntotal > 0:
            query_vector = dummy_embeddings[0] # Search with the first embedding
            top_k = 3
            search_results = search_faiss_index(index, query_vector, top_k)
            if search_results:
                D, I = search_results
                print(f"Search Results for vector 0 (top {top_k}):")
                print(f"  Distances: {D}")
                print(f"  Indices:   {I}")
        else:
            print("Skipping search as no embeddings were added to the index.")

        # 4. Save the index
        print("\n4. Saving the index...")
        save_faiss_index(index, dummy_index_file)

        # 5. Load the index
        print("\n5. Loading the index...")
        loaded_index = load_faiss_index(dummy_index_file)
        if loaded_index:
            print(f"Loaded index. Dimension: {loaded_index.d}, Total: {loaded_index.ntotal}")

        # Cleanup
        if os.path.exists(dummy_index_file):
            print(f"\nCleaning up dummy index file: {dummy_index_file}")
            os.remove(dummy_index_file)
    else:
        print("FAISS index could not be initialized (likely FAISS library not available). Skipping further demonstrations.")

    print("\n--- End of Demonstration ---")
