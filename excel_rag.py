from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def load_docs():
    loader=UnstructuredExcelLoader(file_path="./RohitSharma/Rohit_Sharma_Centuries.xlsx",mode="elements")
    docs=loader.load()
    return docs

def filter_metadata(docs):
    for doc in docs:
        filtered={}
        for key,value in doc.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                filtered[key] = value
            else :
                filtered[key]=str(value) if isinstance(value,list) else None
    
        doc.metadata=filtered
    return docs

def split_docs(docs,chunk_size=512,chunk_overlap=25):
    splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    texts=splitter.split_documents(docs)
    return texts

def get_embeddingmodel():
    model=HuggingFaceEmbeddings(model_name="thenlper/gte-large",model_kwargs={'device':'cpu'})
    return model

def create_VectorDB(texts,embeddings,file_path):
    flag=True
    try:
        db=Chroma.from_documents(texts,embeddings,persist_directory=file_path)
    except Exception as e:
        print("Encountered an exception ",e)
        flag=False 
    return flag
        
def ingest(persist_path="./Cricket"):
    docs=load_docs()
    docs=filter_metadata(docs)
    texts=split_docs(docs)
    emb_model=get_embeddingmodel()
    
    flag=create_VectorDB(texts,emb_model,persist_path)
    if flag:
        print("Vector DB created")
    else :
        print("Soemthing went wrong")
        
    
    
def getRetreiver():
    emb_model=HuggingFaceEmbeddings(model_name="thenlper/gte-large",model_kwargs={'device':'cpu'})
    db=Chroma(persist_directory="./Cricket",embedding_function=emb_model)
    return db

template = """<s>[INST] Given the context - {context} </s>[INST] [INST] Answer the following question - {question}[/INST]"""


def create_prompt():
    prompt=PromptTemplate(template=template,input_variables=['context','question'])
    return prompt

def qna_bot():
    db=getRetreiver()
    retreiver=db.as_retriever(search_kwargs={"k":5})
    
    llm=Ollama(model="gemma2:2b")
    print("LLM loaded ",llm)
    
    chain=(
        {"context": retreiver, "question": RunnablePassthrough()}
        | create_prompt()
        | llm 
        | StrOutputParser()
    )
    
    while True:
        query=input("Your query: ")
        if query=="quit":
            break
        output=chain.invoke(query)
        print(output)
        
    print("You just used a custom cricket bot")


if __name__=="__main__":
    ingest()
    qna_bot()