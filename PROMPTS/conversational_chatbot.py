# Import ChatGroq LLM wrapper from LangChain
from langchain_groq import ChatGroq

# Import message classes used to structure conversation history
from langchain_core.messages import SystemMessage , HumanMessage , AIMessage

# Import dotenv to load environment variables from .env file
from dotenv import load_dotenv

# Import os module to access environment variables
import os

# Load variables from the .env file (like GROQ_API_KEY)
load_dotenv()


# Initialize the Groq LLM model
llm = ChatGroq(
    model="llama-3.1-8b-instant",   # Model name hosted on Groq
    temperature=0.5,                # Controls randomness (0 = deterministic, 1 = creative)
    api_key=os.getenv("GROQ_API_KEY")  # Fetch API key from environment variable
)


# Initialize chat history with a system message
# System message defines the behavior of the AI assistant
chat_history = [
    SystemMessage(content='You are a helpful AI assistant'),
]


# Infinite loop to keep chatting until the user types 'exit'
while True:
    
    # Take input from the user
    user_input = input('You: ')
    
    # Add the user's message to chat history
    chat_history.append(HumanMessage(content=user_input))
    
    # If user types 'exit', stop the chat
    if user_input == 'exit':
        break
    
    # Send the entire conversation history to the model
    # This allows the model to maintain context
    result = llm.invoke(chat_history)
    
    # Add AI's response to the chat history
    chat_history.append(AIMessage(content=result.content))
    
    # Print the AI response
    print("AI : ", result.content)
    

# After exiting the loop, print the full chat history
# This will show all system, user, and AI messages
print(chat_history)




# Loads your Groq API key from .env.

# Initializes the Llama-3.1 model via Groq.

# Maintains a chat history list containing:

# SystemMessage

# HumanMessage

# AIMessage

# Sends the entire history to the model every time so it remembers context.

# Stops when the user types exit.
