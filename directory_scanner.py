import os

def scan_directory(path):
    """
    Recursively scans a directory and returns a list of dictionaries,
    each representing a file or directory with its path and type.
    """
    results = []
    # Error handling: Check if path exists and is a directory
    if not os.path.exists(path):
        raise FileNotFoundError(f"Error: Path '{path}' does not exist.")
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Error: Path '{path}' is not a directory.")

    for item in os.listdir(path):
        item_path = os.path.abspath(os.path.join(path, item))
        if os.path.isdir(item_path):
            results.append({"path": item_path, "type": "directory"})
            # Recursively scan subdirectories
            results.extend(scan_directory(item_path))
        elif os.path.isfile(item_path):
            results.append({"path": item_path, "type": "file"})
    return results

if __name__ == "__main__":
    # Example Usage
    sample_path = "."  # Scan the current directory

    print(f"Scanning directory: {os.path.abspath(sample_path)}")
    try:
        scan_results = scan_directory(sample_path)
        if scan_results:
            print("Scan Results:")
            for item in scan_results:
                print(f"  Path: {item['path']}, Type: {item['type']}")
        else:
            print(f"No items found in '{sample_path}' or it's empty.")
    except FileNotFoundError as e:
        print(e)
    except NotADirectoryError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("\n--- Example with a non-existent path ---")
    non_existent_path = "non_existent_directory"
    print(f"Scanning directory: {non_existent_path}")
    try:
        scan_results = scan_directory(non_existent_path)
    except FileNotFoundError as e:
        print(e)
    except NotADirectoryError as e:
        print(e)

    print("\n--- Example with a file path ---")
    # Create a dummy file for this test
    dummy_file_path = "dummy_file.txt"
    with open(dummy_file_path, "w") as f:
        f.write("This is a dummy file.")

    print(f"Scanning directory: {dummy_file_path}")
    try:
        scan_results = scan_directory(dummy_file_path)
    except NotADirectoryError as e:
        print(e)
    except FileNotFoundError as e: # Should not happen here, but good practice
        print(e)
    finally:
        # Clean up the dummy file
        if os.path.exists(dummy_file_path):
            os.remove(dummy_file_path)

    # Example with a specific directory to be created for testing
    test_dir = "my_test_directory"
    print(f"\n--- Example with a specific test directory: {test_dir} ---")
    if not os.path.exists(test_dir):
        os.makedirs(os.path.join(test_dir, "subdir1"))
        os.makedirs(os.path.join(test_dir, "subdir2"))
        with open(os.path.join(test_dir, "file1.txt"), "w") as f:
            f.write("content1")
        with open(os.path.join(test_dir, "subdir1", "file2.txt"), "w") as f:
            f.write("content2")

    try:
        scan_results = scan_directory(test_dir)
        if scan_results:
            print("Scan Results:")
            for item in scan_results:
                print(f"  Path: {item['path']}, Type: {item['type']}")
        else:
            print(f"No items found in '{test_dir}' or it's empty.")
    except FileNotFoundError as e:
        print(e)
    except NotADirectoryError as e:
        print(e)
    finally:
        # Clean up the test directory
        if os.path.exists(test_dir):
            # Need to remove files and subdirs first
            for root, dirs, files in os.walk(test_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(test_dir)
