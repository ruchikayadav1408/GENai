from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel


load_dotenv()


prompt1 = PromptTemplate(
    input_variables=["text"],
    template="Generate short and simple notes from the following text \n {text}."
)


prompt2= PromptTemplate(
    input_variables=["text"],
    template="Generate a 5 short question answers from the following text \n  {text}."
)

prompt3= PromptTemplate(
    input_variables=["notes", "quiz"],
    template="Merge the provided notes and quiz into a single document \n notes ->  {notes} and quiz ->{quiz}."
)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    # groq_api_key=os.getenv("GROQ_API_KEY")
)
llm2= ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

parser= StrOutputParser()

parrallel_chain = RunnableParallel(
    {
        "notes": prompt1 | llm | parser,
        "quiz": prompt2 | llm2 | parser
    }
)


mergechain= prompt3 | llm | parser
chain=parrallel_chain | mergechain
result = chain.invoke({"text":"India is a country in South Asia. It is the second most populous country in the world, with over 1.3 billion people. India has a rich history and culture, with many famous landmarks such as the Taj Mahal and the Red Fort. The country is known for its diverse cuisine, vibrant festivals, and Bollywood film industry. India is also a major player in the global economy, with a rapidly growing technology sector and a large workforce."})   
print(result)
