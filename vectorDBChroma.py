import os
from typing import Iterable, List, Optional
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import numpy as np
from excelutility import getExcelDatainJson

from vectorDbbase import Embeddings,baseVebtorDb,basevectorDBLangchain

    
class chromaRag(basevectorDBLangchain):
    
    PERCISTENT_DIRECTORY="./indexes/croma/"
    
    def __init__(self, persist_dir=PERCISTENT_DIRECTORY,embedder=Embeddings.getdefaultEmbading()) -> None:
        super().__init__( persist_dir=persist_dir,embedder=embedder)  
       

    def getLangChainRetriver(self):
        
        # return  self.faiss_vector_store.as_retriever() 
        # print(self.faiss_vector_store.embedding_function.model)
        return  self.vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 2,"score_threshold": 0.01})
    

    
            
    
    def save(self,persist_directory=PERCISTENT_DIRECTORY):
        self.vectorstore.persist()
       
        
    # Load FAISS index from disk
    def load(self,persist_directory=PERCISTENT_DIRECTORY):
        self.vectorstore =Chroma(persist_directory=persist_directory, embedding_function=self.embedder)
        
    
            
    def build_data(self,documents):
        self.vectorstore  = Chroma.from_documents(documents, self.embedder, persist_directory=self.PERCISTENT_DIRECTORY)

        # self.faiss_vector_store.embeddings=self.embedder
        
    def add_data(self,documents):
        if self.vectorstore is None:
            self.vectorstore=Chroma.from_documents(documents,self.embedder)
        else:
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            self.add_text(texts,metadatas)
            
            
        # self.faiss_vector_store.embeddings=self.embedder
        
    def add_text(self,texts:Iterable[str],metadatas:Optional[List[dict]] = None):
        if self.vectorstore is None:
            if metadatas:
                documents = [chromaRag.createDocument(page_content=text,metadata=metadata) for text,metadata in zip (texts,metadatas)]
            else:
                documents = [chromaRag.createDocument(page_content=text) for text in texts]
            self.add_data(documents)
        else:
            
            new_texts =[]
            for doc in self.vectorstore.get():
                if not doc['text'] in texts:
                    texts
            existing = [doc['id'] for doc in self.vectorstore.get() if doc['text'] in texts]
            
            new_texts = [text for text in texts if text not in existing_ids]
            # existing_texts = [text for text in texts if text in existing_ids]

            self.vectorstore.add_texts(
                texts=new_texts ,
                metadatas=metadatas
            )
    
    def getVector(self):
        return self.vectorstore
    
    def printDb(self):
        # Assuming 'vector_collections' is your collection object
        all_items = self.vectorstore.get()

        print(all_items)


def check_embeding(ll,query):
    # ll.load()
    from bidi.algorithm import get_display


    print(ll.__class__.__name__,"answer for:\n",get_display(str(query)),"\n",
        get_display(str(ll.data_search(query))))
    
def loadDocumentsfromFaq(faqs):
    import document
    documents=[]
    for item in faqs: 
        document_tmp =basevectorDBLangchain.createDocument(page_content=" question : "+ str( item['question']) +" \n answer : "+ str(item['answer'])+ "\n",
            metadata= "question"+ item['question']
            
        )
        documents.append(document_tmp)   
        
    return documents    
        
      

def loadChromaDB(faqs,myembeder):
    ll=chromaRag(embedder=myembeder)
    ll.delete()
    documents=loadDocumentsfromFaq(faqs)
    ll.build_data(documents)
    ll.save() 
    
def loadChromaDBfromExcel(chromaRag):
    json_data =  getExcelDatainJson()
    print("load additional questions to chroma")
    documents=loadDocumentsfromFaq(json_data) 
    chromaRag.add_data(documents)    
    # chromaRag.printDb()
    return chromaRag  

def getVectorDB(embeding):
    ll=chromaRag(embedder=embeding)
    return loadChromaDBfromExcel(ll)    
    
if __name__ == "__main__":         

    myembeder=Embeddings.getdefaultEmbading()

    # loadChromaDB(document.faqs,myembeder)
    ll=chromaRag(embedder=myembeder)

    query = "איך אני מקבל תמיכה?"

    check_embeding(ll,query)
    
    query = "מיהכן אוספים מטח?"

    check_embeding(ll,query)