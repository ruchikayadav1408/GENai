//simple with_structured_output

from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from typing import TypedDict
load_dotenv()


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5,
    api_key=os.getenv("GROQ_API_KEY")
)

class Review(TypedDict):
    summary: str
    sentiment: str
    
    
    
structed_model=llm.with_structured_output(Review)
result = structed_model.invoke("""The hardware is great , but the software feels bloadted.There are too many pre=installed apps that I cant remove.
                    Also . the UI looks outdated compared to other brands. Hoping for a software update to fix this.""")


print(result)
