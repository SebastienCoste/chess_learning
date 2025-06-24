import os
from pathlib import Path

def concatenate_py_files(source_folder, output_file):
    """
    Concatenate all .py files from source_folder (recursively) into a single output file.
    Each file's content is separated by a header with its original path.
    """
    source_path = Path(source_folder)
    py_files = sorted(source_path.rglob("*.py"))  # Sorted for consistent order

    with open(output_file, "w", encoding="utf-8") as outfile:
        for py_file in py_files:
            header = f"\n\n# --- START OF FILE: {py_file.relative_to(source_path)} ---\n\n"
            outfile.write(header)
            try:
                with open(py_file, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())
            except Exception as e:
                outfile.write(f"# Could not read {py_file}: {e}\n")
            footer = f"\n# --- END OF FILE: {py_file.relative_to(source_path)} ---\n"
            outfile.write(footer)

    print(f"All .py files have been concatenated into '{output_file}'.")

def main():
    source_folder = input("Enter the path to the source folder containing .py files: ").strip()
    output_file = input("Enter the path for the output file (e.g., all_code.txt): ").strip()
    if not os.path.exists(source_folder):
        print(f"Error: Source folder '{source_folder}' does not exist.")
        return
    concatenate_py_files(source_folder, output_file)

if __name__ == "__main__":
    main()
