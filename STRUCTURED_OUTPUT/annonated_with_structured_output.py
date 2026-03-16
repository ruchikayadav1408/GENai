from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from typing import TypedDict, Annotated, Optional

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5,
    api_key=os.getenv("GROQ_API_KEY")
)

class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]

    summary: Annotated[str, "Write a detailed 5-6 line summary explaining the full review in detail"]

    sentiment: Annotated[str, "Return sentiment of the review either negative, positive or neutral"]

    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]

    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]


structured_model = llm.with_structured_output(Review)

result = structured_model.invoke(
"""The hardware is great, but the software feels bloated. 
There are too many pre-installed apps that I can't remove. 
Also, the UI looks outdated compared to other brands. 
Hoping for a software update to fix this."""
)

print(result)



# {
#  'key_themes': ['hardware quality', 'bloated software', 'pre-installed apps', 'outdated UI', 'need for software update'],
#  'summary': 'The reviewer appreciates the strong hardware quality of the device but expresses dissatisfaction with the software experience. They feel the system contains too many pre-installed applications that cannot be removed, which contributes to a bloated interface. Additionally, the user interface design appears outdated when compared with competing brands. Overall, the review highlights a mismatch between excellent hardware and weak software optimization. The reviewer hopes that a future software update will resolve these issues.',
#  'sentiment': 'negative',
#  'pros': ['great hardware'],
#  'cons': ['bloated software', 'too many pre-installed apps', 'outdated UI']
# }
