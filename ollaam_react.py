from langchain.agents import AgentExecutor,create_react_agent
from langchain_community.tools import TavilySearchResults
from langchain_community.llms import Ollama
from langchain import hub

from dotenv import load_dotenv
import os

load_dotenv()

llm=Ollama(model="gemma2:2b")

query="What happened in the Barca Villareal game in September 2024"

tools=[TavilySearchResults(max_results=2)]

prompt=hub.pull('hwchase17/react')

agent=create_react_agent(llm,tools,prompt)

agent_exec=AgentExecutor(agent=agent,tools=tools,verbose=True)

agent_exec.invoke({"input":query})

