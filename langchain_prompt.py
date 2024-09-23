from langchain.prompts import PromptTemplate


template = """
What do you think of 
{subject}
"""


prompt = PromptTemplate(
    input_variables=["subject"],
    template=template,
)


formatted_prompt = prompt.format(subject="Tyrion Lannister")
print(formatted_prompt)
