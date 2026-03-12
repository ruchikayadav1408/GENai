# --------------------- Static Prompt Research Tool ---------------------

# Import Streamlit to build the web interface
import streamlit as st

# Import ChatGroq to access Groq LLM models through LangChain
from langchain_groq import ChatGroq

# Import dotenv to load environment variables from a .env file
from dotenv import load_dotenv

# Import os module to read environment variables
import os


# Load environment variables (like GROQ_API_KEY) from .env file
load_dotenv()


# Display the title/header of the Streamlit app
st.header("Research Tool")


# Create an LLM object using Groq
llm = ChatGroq(
    model="llama-3.1-8b-instant",     # Fast and efficient model provided by :contentReference[oaicite:0]{index=0}
    temperature=0.5,                  # Controls randomness (lower = more factual, higher = more creative)
    api_key=os.getenv("GROQ_API_KEY") # Fetch API key from environment variable
)


# Create a text input box where the user can type a prompt
user_input = st.text_input("Enter your prompt")


# Create a button to trigger the model
if st.button("Generate"):

    # Check if user entered something
    if user_input:

        # Send the prompt to the LLM and get the response
        result = llm.invoke(user_input)

        # Display the AI-generated response on the Streamlit page
        st.write(result.content)
