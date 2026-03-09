from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables (OPENAI_API_KEY)
load_dotenv()

# Initialize OpenAI embedding model
embedding = OpenAIEmbeddings(
    model='text-embedding-3-large',
    dimensions=300
)

# Documents related to cricket players
documents = [
    "Virat Kohli is one of the greatest batsmen in modern cricket and former captain of the Indian team.",
    "Jasprit Bumrah is India's leading fast bowler known for his deadly yorkers.",
    "Rohit Sharma is the captain of the Indian cricket team and famous for scoring big centuries.",
    "MS Dhoni is a legendary wicketkeeper batsman and led India to multiple ICC trophies.",
    "Sachin Tendulkar is known as the God of Cricket and scored 100 international centuries."
]

# Query
query = "tell me about virat kohli"

# Convert documents into embeddings
doc_embedding = embedding.embed_documents(documents)

# Convert query into embedding
query_embedding = embedding.embed_query(query)

# Compute cosine similarity between query and documents
scores = cosine_similarity([query_embedding], doc_embedding)[0]

# Get the most similar document
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

# Print result
print("Query:", query)
print("Most Similar Document:", documents[index])
print("Similarity score is:", score)



# Convert documents → embeddings

# Convert query → embedding

# Use cosine similarity to compare

# Return the most relevant document
