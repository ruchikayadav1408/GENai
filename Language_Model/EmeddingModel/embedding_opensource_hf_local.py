# Import HuggingFaceEmbeddings from langchain_huggingface
# This class helps generate vector embeddings using HuggingFace models

# Loads a Sentence Transformer embedding model.

# Takes a text sentence.

# Converts it into a 384-dimensional numerical vector.

# Prints the vector.

# Example output (shortened):

# [0.0123, -0.0345, 0.0678, ...]
from langchain_huggingface import HuggingFaceEmbeddings


# Initialize the embedding model
# "sentence-transformers/all-MiniLM-L6-v2" is a lightweight and fast embedding model
# It converts text into a numerical vector representation
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Sample text that we want to convert into an embedding vector
text = "Delhi is the capital of India"


# Generate the embedding vector for the given text
# embed_query() converts the text into a list of numbers (vector)
vector = embedding.embed_query(text)


# Print the vector as a string
# The vector represents the semantic meaning of the sentence
print(str(vector))
