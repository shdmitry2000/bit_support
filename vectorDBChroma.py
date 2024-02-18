import os
from typing import List
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import numpy as np

from vectorDbbase import Embeddings,baseVebtorDb

    
class chromaRag(baseVebtorDb):
    
    PERCISTENT_DIRECTORY="./indexes/croma/"
    
    def __init__(self, persist_dir=PERCISTENT_DIRECTORY,embedder=Embeddings.getdefaultEmbading()) -> None:
        super().__init__( persist_dir=persist_dir,embedder=embedder)  
       

    def getLangChainRetriver(self):
        
        # return  self.faiss_vector_store.as_retriever() 
        # print(self.faiss_vector_store.embedding_function.model)
        return  self.vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 2,"score_threshold": 0.01})
    

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
        self.vectorstore.persist()
       
        
    # Load FAISS index from disk
    def load(self,persist_directory=PERCISTENT_DIRECTORY):
        self.vectorstore =Chroma(persist_directory=persist_directory, embedding_function=self.embedder)
        
    def delete(self,path="./indexes/chroma/"):
        import shutil
        if os.path.exists(path):
            # Use shutil.rmtree() to delete the folder and all its contents
            shutil.rmtree(path)
            
    def build_data(self,documents):
        self.vectorstore  = Chroma.from_documents(documents, self.embedder, persist_directory=self.PERCISTENT_DIRECTORY)

        # self.faiss_vector_store.embeddings=self.embedder
        
    def add_data(self,documents):
        
        if self.vectorstore is None:
            self.vectorstore=Chroma.from_documents(documents,self.embedder)
        else:
            self.vectorstore.from_documents(documents,self.embedder)
            
        # self.faiss_vector_store.embeddings=self.embedder
        
    def add_text(self,texts):
        if self.vectorstore is None:
            documents = [chromaRag.createDocument(page_content=text) for text in texts]
            # print(documents)
            self.vectorstore=Chroma.from_documents(documents,self.embedder)
        else:
            self.vectorstore.add_texts(texts)    
        
    


def check_embeding(ll,query):
    # ll.load()
    from bidi.algorithm import get_display


    print(ll.__class__.__name__,"answer for:\n",get_display(str(query)),"\n",
        get_display(str(ll.data_search(query))))
    
def buildDB():
    ll=chromaRag(embedder=myembeder)
    ll.delete()
    import document
    documents=[]
    for item in document.faqs: 
        document_tmp =ll.createDocument(page_content=" question : "+ str( item['question']) +" \n answer : "+ str(item['answer'])+ "\n",
            metadata= "question"+ item['question']
            
        )
        documents.append(document_tmp)    
         
    ll.build_data(documents)
    ll.build_data(documents=document.get_documents())
    ll.save() 
    
if __name__ == "__main__":         

    myembeder=Embeddings.getdefaultEmbading()

    # buildDB()
    

    query = "איך אני מקבל תמיכה?"

    check_embeding(chromaRag(embedder=myembeder),query)
    
    query = "מיהכן אוספים מטח?"

    check_embeding(chromaRag(embedder=myembeder),query)