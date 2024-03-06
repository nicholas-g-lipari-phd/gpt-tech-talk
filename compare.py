from zipfile import ZipFile
import os

# Define the path for the uploaded ZIP file and the extraction directory
zip_file_path = '/workspaces/gpt-tech-talk/data/data.zip'
extraction_path = '/workspaces/gpt-tech-talk/data/extracted/'

# Create a directory for extracted files
os.makedirs(extraction_path, exist_ok=True)

# Extract the ZIP file
with ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_path)

# List the extracted contents
extracted_files = []
for root, dirs, files in os.walk(extraction_path):
    for file in files:
        extracted_files.append(os.path.relpath(os.path.join(root, file), extraction_path))

print(extracted_files)
