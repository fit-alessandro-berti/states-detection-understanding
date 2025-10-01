import os

def merge_python_files(folder_path, output_file="merged_python_files.txt"):
    # Get all .py files in the folder
    py_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".py")])

    with open(output_file, "w", encoding="utf-8") as outfile:
        for file_name in py_files:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as infile:
                outfile.write(f"# --- Start of {file_name} ---\n")
                outfile.write(infile.read())
                outfile.write(f"\n# --- End of {file_name} ---\n\n")

    print(f"✅ Merged {len(py_files)} files into {output_file}")


if __name__ == "__main__":
    merge_python_files("..")
