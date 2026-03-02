"""
LangChain + HuggingFace Hub (Remote Inference) Example
------------------------------------------------------

This script:
1. Loads environment variables from a .env file.
2. Connects to a HuggingFace hosted model using API token.
3. Wraps it inside ChatHuggingFace for conversational format.
4. Sends a prompt and prints the response.

Model Used:
TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

# Import Chat wrapper and HuggingFace endpoint connector
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Used to load environment variables from .env file
from dotenv import load_dotenv

# Used to access environment variables securely
import os


# Step 1: Load environment variables
# -----------------------------------
# This loads variables from .env into the system environment.
# Make sure your .env file contains:
# HUGGINGFACEHUB_API_TOKEN=your_actual_token_here
load_dotenv()


# Step 2: Initialize HuggingFace Hub endpoint
# --------------------------------------------
# repo_id → Model name from HuggingFace Hub
# task → Type of model task (text-generation)
# huggingfacehub_api_token → Authentication token
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)


# Step 3: Wrap LLM into Chat format
# ----------------------------------
# ChatHuggingFace provides conversational interface
model = ChatHuggingFace(llm=llm)


# Step 4: Send a prompt to the model
# -----------------------------------
response = model.invoke("What is the capital of India?")


# Step 5: Print generated response
# ---------------------------------
# response object contains metadata + content
# We print only the generated text
print(response.content)
