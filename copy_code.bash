#!/usr/bin/env python3
import os

output_file = "all_py_files.txt"
py_files = []

# Collect all .py files
for root, dirs, files in os.walk("."):
    for name in files:
        if name.endswith(".py"):
            file_path = os.path.join(root, name)
            py_files.append(file_path)

# Write to output file
with open(output_file, "w", encoding="utf-8") as outfile:
    # Table of contents (tree format)
    outfile.write("TABLE OF CONTENTS (Python Files)\n")
    outfile.write("=" * 50 + "\n")
    for f in py_files:
        # Indent based on folder depth
        depth = f.count(os.sep) - 1  # relative to current folder
        outfile.write("  " * depth + f"- {f}\n")
    outfile.write("\n\n")

    # Concatenate file contents
    for f in py_files:
        outfile.write(f"===== FILE: {f} =====\n")
        try:
            with open(f, "r", encoding="utf-8") as src:
                outfile.write(src.read())
        except Exception as e:
            outfile.write(f"\n[Error reading file: {e}]\n")
        outfile.write(f"\n===== END OF FILE: {f} =====\n\n")

print(f"Created {output_file} with TOC and all .py files.")