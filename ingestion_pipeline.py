import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path='docs'):
    print("Here is details of document from data source")
    print(f"loading documents from {docs_path}...")
    
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. please create it ")
    loader=DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}  
    ) 
    
    
    documents=loader.load()
    
    if len(documents)==0:
        raise FileNotFoundError(f"No, txt files found in {docs_path}.please add your company documents")
    
    for i, doc in enumerate(documents[:len(documents)]):
        print(f"\n Documents {i+1}")
        print(f"Source:{doc.metadata['source']}")
        print(f"Content length:{len(doc.page_content)} characters")
        print(f"Content preview:{doc.page_content[:100]}...")
        print(f"metadata:{doc.metadata}")
    return documents   

def split_documents(documents,chunk_size=800,chunk_overlap=0):
    print("Splitting documents into chunks...")
    
    text_splitter=CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks=text_splitter.split_documents(documents)
    
    if chunks:
        for i,chunk in enumerate(chunks[:5]):
            print(f"\n--Chunk {i+1}---")
            print(f"Source:{chunk.metadata['source']}")
            print(f"Length {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-"*50)
            
        if len(chunks)>5:
            print(f"\n.. and {len(chunks)-5} more chunks")    
    return chunks        

documents=load_documents(docs_path="docs") 
chunks=split_documents(documents)  

  
    

