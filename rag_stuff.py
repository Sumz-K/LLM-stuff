from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def load_docs():
    loader=DirectoryLoader('./pdfs',glob="*.pdf",use_multithreading=True)
    docs=loader.load()
    return docs

def split_docs(docs,chunk_size=512,chunk_overlap=40):
    splitter= RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    texts=splitter.split_documents(docs)
    return texts


def get_embedding_model():
    emb_model=HuggingFaceEmbeddings(model_name="thenlper/gte-large",model_kwargs={'device':'cpu'})
    return emb_model


def createVectordb(texts,embeddings,persist_path):
    flag=True
    try:
        vector_db=Chroma.from_documents(texts,embeddings,persist_directory=persist_path)
    except Exception as e: 
        flag = False
        print(f"Error creating vector database: {str(e)}") 
        
    return flag


def ingest():
    docs=load_docs()
    texts=split_docs(docs)
    embedding_model=get_embedding_model()
    
    flag=createVectordb(texts,embedding_model,"./chroma")
    if flag:
        print("Vector db created")
    else:
        print("Something went wrong")
        
        


def getRetreiver():
    emb_model=HuggingFaceEmbeddings(model_name="thenlper/gte-large",model_kwargs={'device':'cpu'})
    db=Chroma(persist_directory="./chroma",embedding_function=emb_model)
    return db
    
template = """<s>[INST] Given the context - {context} </s>[INST] [INST] Answer the following question - {question}[/INST]"""


def create_prompt():
    prompt=PromptTemplate(template=template,input_variables=['context','question'])
    return prompt


def qna_bot():
    db=getRetreiver()
    retreiver=db.as_retriever(search_kwargs={"k":2})
    
    llm=Ollama(model="gemma2:2b")
    print("LLM loaded ",llm)
    
    chain=(
        {"context": retreiver, "question": RunnablePassthrough()}
        | create_prompt()
        | llm 
        | StrOutputParser()
    )
    
    query=""
    
    while True:
        query=input("Your query: ")
        if query=="quit":
            break
        output=chain.invoke(query)
        print(output)
        
    print("You just used a custombot")
        
    
if __name__=="__main__":
    ingest()
    qna_bot()