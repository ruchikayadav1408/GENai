from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


prompt = PromptTemplate(
    input_variables=["topic"],
    template="Generate 5 interesting facts aboit {topic}."
)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    # groq_api_key=os.getenv("GROQ_API_KEY")
)


parser= StrOutputParser()

chain = prompt | llm | parser
result = chain.invoke({"topic":"India"})
print(result)
