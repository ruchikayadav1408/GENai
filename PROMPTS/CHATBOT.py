# Import ChatGroq class to interact with Groq LLM API
from langchain_groq import ChatGroq

# Import dotenv to load environment variables from .env file
from dotenv import load_dotenv

# Import os module to access environment variables
import os


# Load variables from the .env file (like GROQ_API_KEY)
load_dotenv()


# Create an instance of the Groq language model
llm = ChatGroq(
    model="llama-3.1-8b-instant",     # Model name provided by :contentReference[oaicite:0]{index=0}
    temperature=0.5,                  # Controls randomness (0 = deterministic, 1 = more creative)
    api_key=os.getenv("GROQ_API_KEY") # Fetch API key from environment variable
)


# Start an infinite loop to simulate a chatbot conversation
while True:

    # Take input from the user
    user_input = input('You: ')

    # If the user types 'exit', break the loop and stop the program
    if user_input == 'exit':
        break

    # Send the user's message to the LLM and get the response
    result = llm.invoke(user_input)

    # Print the AI's response
    print("AI :", result.content)



You: What is AI?
AI: Artificial Intelligence (AI) refers to systems that can perform tasks
that normally require human intelligence such as reasoning, learning,
problem solving, and language understanding.

You: explain transformer model
AI: The transformer is a neural network architecture introduced in
"Attention Is All You Need" that relies on self-attention mechanisms.

You: exit
