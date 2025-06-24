import os
import shutil
from pathlib import Path


def flatten_py_to_txt(source_folder, destination_folder):
    """
    Converts all .py files in source_folder (including subfolders) to .txt files
    and places them in a flattened structure in destination_folder.

    Args:
        source_folder (str): Path to the source folder containing .py files
        destination_folder (str): Path to the destination folder for .txt files
    """

    # Create destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Convert to Path objects for easier manipulation
    source_path = Path(source_folder)
    dest_path = Path(destination_folder)

    # Counter for handling duplicate filenames
    file_counter = {}

    # Walk through all files in source folder and subfolders
    for py_file in source_path.rglob("*.py"):
        # Get the filename without extension
        base_name = py_file.stem

        # Handle duplicate filenames by adding a counter
        if base_name in file_counter:
            file_counter[base_name] += 1
            txt_filename = f"{base_name}_{file_counter[base_name]}.txt"
        else:
            file_counter[base_name] = 0
            txt_filename = f"{base_name}.txt"

        # Create destination file path
        dest_file = dest_path / txt_filename

        try:
            # Copy the file content from .py to .txt
            shutil.copy2(py_file, dest_file)
            print(f"Converted: {py_file} -> {dest_file}")
        except Exception as e:
            print(f"Error converting {py_file}: {e}")


def main():
    # Get input from user
    source_folder = input("Enter the path to the source folder containing .py files: ").strip()
    destination_folder = input("Enter the path to the destination folder for .txt files: ").strip()

    # Validate source folder exists
    if not os.path.exists(source_folder):
        print(f"Error: Source folder '{source_folder}' does not exist.")
        return

    # Convert files
    print(f"\nConverting .py files from '{source_folder}' to .txt files in '{destination_folder}'...")
    flatten_py_to_txt(source_folder, destination_folder)
    print("\nConversion completed!")


if __name__ == "__main__":
    main()
