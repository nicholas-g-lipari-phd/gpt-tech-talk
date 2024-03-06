from zipfile import ZipFile
import os
from collections import defaultdict
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Define the path for the uploaded ZIP file and the extraction directory
zip_file_path = '/workspaces/gpt-tech-talk/data/data.zip'
extraction_path = '/workspaces/gpt-tech-talk/data/extracted/'

def read_zip():
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

    return extracted_files

# Function to read and preprocess Java files
def read_preprocess_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    # Remove comments and whitespace for better similarity comparison
    content = re.sub(r'//.*?\n|/\*.*?\*/', '', content, flags=re.DOTALL)
    content = re.sub(r'\s+', ' ', content).strip()
    return content

extracted_files = read_zip()

# Prepare data for TF-IDF Vectorizer
java_programs = {file_path: read_preprocess_file(os.path.join(extraction_path, file_path)) for file_path in extracted_files}

# Instantiate the vectorizer
vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(), lowercase=False)

# Tokenize and transform the data into TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(java_programs.values())

# Calculate cosine similarity matrix
cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Prepare the similarity scores excluding self-comparison
similarity_scores = defaultdict(dict)
for i, file_path_i in enumerate(java_programs.keys()):
    for j, file_path_j in enumerate(java_programs.keys()):
        if i != j:
            similarity_scores[file_path_i][file_path_j] = cosine_sim_matrix[i, j]

# Organize the results in a more readable way
readable_similarity_scores = defaultdict(list)
for file_i, comparisons in similarity_scores.items():
    for file_j, score in comparisons.items():
        readable_similarity_scores[file_i].append((file_j, score))

# Sort the scores for each file
for file, scores in readable_similarity_scores.items():
    scores.sort(key=lambda x: x[1], reverse=True)

# Take top 3 similar for each file for demonstration
top_3_similar = {file: scores[:3] for file, scores in readable_similarity_scores.items()}

print(top_3_similar)

