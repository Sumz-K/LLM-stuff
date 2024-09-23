from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

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
        
    return flag,vector_db


def ingest():
    docs=load_docs()
    texts=split_docs(docs)
    embedding_model=get_embedding_model()
    
    flag,db=createVectordb(texts,embedding_model,"./chroma")
    if flag:
        print("Vector db created")
    else:
        print("Something went wrong")
        
    testQuery(db)
        


def testQuery(db):
    query="Who are the authors of DeepRED"
    ans=db.similarity_search(query)
    print(ans[0].page_content)
    

if __name__=="__main__":
    ingest()