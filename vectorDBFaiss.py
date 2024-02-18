import os
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import numpy as np
from vectorDbbase import Embeddings,baseVebtorDb


 
class faissRag(baseVebtorDb):
    PERCISTENT_DIRECTORY="./indexes/faiss/"
    
    def __init__(self, persist_dir=PERCISTENT_DIRECTORY,embedder=Embeddings.getdefaultEmbading()) -> None:
        super().__init__( persist_dir=persist_dir,embedder=embedder)  
       
    def getLangChainRetriver(self):
        
        # return  self.vectorstore.as_retriever() 
        # print(self.vectorstore.embedding_function.model)
        return  self.vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 2,"score_threshold": 0.2})

    @staticmethod
    def createDocument(page_content,metadata=""):
        from langchain_core.documents import Document
        
        return Document(
            page_content=page_content,
            metadata=
            {
                "question": metadata,
            }
        )    
        
        
            
    
    def save(self,persist_directory=PERCISTENT_DIRECTORY):
 
        self.vectorstore.save_local(persist_directory)
        
    # Load FAISS index from disk
    def load(self,persist_directory=PERCISTENT_DIRECTORY):
        # Load the vector store from the same directory
        self.vectorstore = FAISS.load_local(persist_directory, embeddings=self.embedder)
        # print(self.vectorstore)

    
    def build_data(self,documents):
        
        self.vectorstore = FAISS.from_documents(documents,self.embedder)
        # self.vectorstore.embeddings=self.embedder
        
    def add_data(self,documents):
        
        if self.vectorstore is None:
            self.vectorstore=FAISS.from_documents(documents,self.embedder)
        else:
            self.vectorstore.from_documents(documents,self.embedder)
            
        # self.vectorstore.embeddings=self.embedder
        
    def add_text(self,texts):
        documents = [faissRag.createDocument(page_content=text) for text in texts]
        self.add_data(documents)
        
    

def check_embeding(ll,query):
    # ll.load()
    from bidi.algorithm import get_display


    print(ll.__class__.__name__,"answer for:\n",get_display(str(query)),"\n",
        get_display(str(ll.data_search(query))))

def loadDB():
    ll=faissRag(embedder=myembeder)
    ll.delete()
    import document
    documents=[]
    for item in document.faqs: 
        document_tmp =ll.createDocument(page_content=" question : "+ str( item['question']) +" \n answer : "+ str(item['answer'])+ "\n",
            metadata={
                "question": item['question'],
            }
        )
        documents.append(document_tmp)    
         
    ll.build_data(documents)
    ll.build_data(documents=document.get_documents())
    ll.save() 
if __name__ == "__main__":         

    myembeder=Embeddings.getdefaultEmbading()

    # loadDB()
    
    

    query = "איך אני מקבל תמיכה?"

    check_embeding(faissRag(embedder=myembeder),query)
    
    query = "מיהכן אוספים מטח?"

    check_embeding(faissRag(embedder=myembeder),query)