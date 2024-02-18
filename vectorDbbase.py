from abc import abstractmethod
import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import numpy as np


load_dotenv()

class Embeddings():

    @staticmethod  
    def getdefaultEmbading():
        return Embeddings.getOpenAi3LargeEmbadings()
    
    @staticmethod       
    def getOpenAiEmbadings():
        return OpenAIEmbeddings(model="text-embedding-ada-002")
    
    @staticmethod       
    def getOpenAi3SmallEmbadings():
        return OpenAIEmbeddings(model="text-embedding-3-small")
    
    @staticmethod       
    def getOpenAi3LargeEmbadings():
        return OpenAIEmbeddings(model="text-embedding-3-large")
    
    



# documents=[]
# document_tmp = createDocument(metadata='question',page_content= " i am happy")
# documents.append(document_tmp) 

# db_openAIEmbedd = FAISS.from_documents(documents, Embeddings.getdefaultEmbading())
# retriever_openai = db_openAIEmbedd.as_retriever(search_kwargs={"k": 3})


    
class baseVebtorDb():
    
   
    
    def __init__(self, persist_dir,embedder=Embeddings.getdefaultEmbading()) -> None:
        self.embedder=embedder
        if os.path.exists(f"{persist_dir}"):
            self.load(persist_directory=persist_dir)
        else:
            self.vectorstore=None 
            
        
        
           
    @abstractmethod
    def getLangChainRetriver(self):
        pass
        
    
    def data_search(self,query: str)-> str:
        
        
        # Get the retriever configured to return source documents
        retriever = self.getLangChainRetriver()
        
        # self.asimilarity_search_with_score_by_vector
        docs = retriever.get_relevant_documents(query,embedding=self.embedder) #,metadatas=metadatas_list
        return docs

    @abstractmethod
    def save(self,persist_directory):
        pass
        
    @abstractmethod
    def load(self,persist_directory):
        pass

    def delete(self,path="./indexes/chroma/"):
        import shutil
        if os.path.exists(path):
            # Use shutil.rmtree() to delete the folder and all its contents
            shutil.rmtree(path)
    
    @abstractmethod     
    def build_data(self,documents):
        pass
    
    @abstractmethod
    def add_data(self,documents):
        pass
    
    @abstractmethod
    def add_text(self,texts):
        if self.vectorstore is None:
            documents = [chromaRag.createDocument(page_content=text) for text in texts]
            # print(documents)
            self.vectorstore=Chroma.from_documents(documents,self.embedder)
        else:
            self.vectorstore.add_texts(texts)    
    
    @abstractmethod
    def printDb(self):
        raise RuntimeError("not implemented yey")

