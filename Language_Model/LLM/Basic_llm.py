"""
LangChain + Groq API Example
----------------------------

This script demonstrates how to:
1. Load environment variables securely using python-dotenv.
2. Connect to Groq's LLM using LangChain.
3. Send a prompt to the model.
4. Print the generated response.

Model Used:
llama-3.1-8b-instant (via Groq API)
"""

# Import required libraries
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os


# Step 1: Load environment variables
# -----------------------------------
# This loads variables from a .env file into the system environment.
# Make sure you have a .env file containing:
# GROQ_API_KEY=your_actual_api_key_here
load_dotenv()


# Step 2: Initialize Groq LLM
# ----------------------------
# model: specifies which Groq-supported model to use
# groq_api_key: securely fetched from environment variable
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)


# Step 3: Send prompt to model
# -----------------------------
# invoke() sends a single query and returns the response object
result = llm.invoke("What is the capital of UP?")


# Step 4: Print model output
# ---------------------------
# result contains metadata + generated content
# We print only the actual text response
print(result.content)
