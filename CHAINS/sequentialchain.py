from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


prompt1 = PromptTemplate(
    input_variables=["topic"],
    template="Generate a detailed report on  {topic}."
)


prompt2= PromptTemplate(
    input_variables=["text"],
    template="Generate a 5 pointer summary from the following text \n  {text}."
)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    # groq_api_key=os.getenv("GROQ_API_KEY")
)


parser= StrOutputParser()

chain = prompt1 | llm | parser | prompt2 | llm | parser
result = chain.invoke({"topic":"Unemployment in India"})
print(result)
