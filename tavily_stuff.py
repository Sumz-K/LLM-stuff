from langchain_community.tools import TavilySearchResults


tool=TavilySearchResults(
    max_results=3,
    include_answer=True
)

response=tool.invoke({'query':'What happened in the Girona vs Barcelona game in September 2024'})

print(response)


