# Import PromptTemplate from LangChain
from langchain_core.prompts import PromptTemplate


# Create a prompt template with placeholders
template = PromptTemplate(

    # Define the prompt text with variables that will be filled later
    template="""
Explain the research paper "{paper_input}" in a {style_input} style.

The explanation should be {length_input}.

Include the following sections:

1. Main idea of the paper
2. Problem the paper solves
3. Key techniques or architecture
4. Why it is important in AI research

Make the explanation clear and structured.
""",

    # List of variables used in the template
    input_variables=['paper_input', 'style_input', 'length_input']
)


# Save the template as a JSON file
# This file can later be loaded using load_prompt("template.json")
template.save('template.json')
