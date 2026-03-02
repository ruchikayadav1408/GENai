"""
LangChain + Groq Example
------------------------

This script:
1. Loads environment variables from a .env file.
2. Initializes a Groq LLM model using LangChain.
3. Sends a prompt to the model.
4. Prints the generated response.

Model Used:
llama-3.3-70b-versatile
"""

# Import Groq chat model from LangChain integration
from langchain_groq import ChatGroq

# Used to load environment variables from .env file
from dotenv import load_dotenv

# Provides access to system environment variables
import os


# Step 1: Load environment variables
# -----------------------------------
# This loads variables (like GROQ_API_KEY) from a .env file
# into the system environment.
load_dotenv()


# Step 2: Initialize Groq LLM
# ----------------------------
# The model parameter specifies which Groq-hosted model to use.
# If groq_api_key is not passed explicitly,
# ChatGroq will automatically read GROQ_API_KEY
# from environment variables.
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    # groq_api_key=os.getenv("GROQ_API_KEY")  # Optional if already set in environment
)


# Step 3: Send a prompt to the model
# -----------------------------------
# invoke() sends a single message to the LLM
# and returns a response object.
result = llm.invoke("Suggest male Indian names?")


# Step 4: Print only the generated text content
# ---------------------------------------------
# The response object contains metadata + content.
# We print only the model's generated text.
print(result.content)
