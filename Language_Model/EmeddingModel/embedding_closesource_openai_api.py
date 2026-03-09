# Import OpenAI embedding class from langchain_openai
# This class is used to generate embeddings using OpenAI models
from langchain_openai import OpenAIEmbeddings

# Import load_dotenv to load environment variables from a .env file
from dotenv import load_dotenv

# Load variables from the .env file (like OPENAI_API_KEY)
load_dotenv()

# Initialize the OpenAI embedding model
# 'text-embedding-3-large' is a powerful embedding model from OpenAI
# dimensions=32 reduces the size of the embedding vector to 32 dimensions
embedding = OpenAIEmbeddings(
    model='text-embedding-3-large',
    dimensions=32
)

# Convert the text query into an embedding vector
# embed_query() returns a list of numerical values representing the text
result = embedding.embed_query("Delhi is capital of india")

# Print the embedding vector as a string
print(str(result))




# Loads your OpenAI API key from the .env file.

# Initializes the OpenAI embedding model (text-embedding-3-large).

# Converts the sentence "Delhi is capital of india" into a numerical vector.

# Prints the resulting embedding vector.

# Example output (shortened):

# [0.021, -0.134, 0.556, ...]
