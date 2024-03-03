import os
from typing import Iterable, List, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import numpy as np
from excelutility import getExcelDatainJson
from vectorDbbase import Embeddings,baseVebtorDb, basevectorDBLangchain


 
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

    def delete(self,path=PERCISTENT_DIRECTORY):
        import shutil
        if os.path.exists(path):
            # Use shutil.rmtree() to delete the folder and all its contents
            shutil.rmtree(path)
            
    def getVector(self):
        return self.vectorstore

    def build_data(self,documents):
        
        self.vectorstore = FAISS.from_documents(documents,self.embedder)
        # self.vectorstore.embeddings=self.embedder
        
    def add_data(self,documents):
        if self.vectorstore is None:
            self.vectorstore=FAISS.from_documents(documents,self.embedder)
        else:
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            self.add_text(texts,metadatas)
            
            
        # self.faiss_vector_store.embeddings=self.embedder
        
    def add_text(self,texts:Iterable[str],metadatas:Optional[List[dict]] = None):
        if self.vectorstore is None:
            if metadatas:
                documents = [faissRag.createDocument(page_content=text,metadata=metadata) for text,metadata in zip (texts,metadatas)]
            else:
                documents = [faissRag.createDocument(page_content=text) for text in texts]
            self.add_data(documents)
        else:
            self.vectorstore.add_texts(
                texts=texts ,
                metadatas=metadatas
            )
    
    def getVector(self):
        return self.vectorstore
    
        
    def printDb(self):
        # Assuming 'index' is your FAISS index and 'docs' is a dictionary mapping IDs to documents
        for id, doc in self.vectorstore.items():
            print(f"ID: {id}, Document: {doc}")


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
        
      

def loadfaissDB(faqs,myembeder):
    ll=faissRag(embedder=myembeder)
    ll.delete()
    documents=loadDocumentsfromFaq(faqs)
    ll.build_data(documents)
    ll.save() 
    
def loadfaissDBfromExcel(faissRag):
    json_data =  getExcelDatainJson()
    print("load additional questions to feiss")
    documents=loadDocumentsfromFaq(json_data) 
    faissRag.add_data(documents)    
    # faissRag.printDb()
    return faissRag  

def getVectorDB(embeding):
    ll=faissRag(embedder=embeding)
    return loadfaissDBfromExcel(ll)   

 
if __name__ == "__main__":         

    myembeder=Embeddings.getdefaultEmbading()

    # loadfaissDB(document.faqs,myembeder)
    
    ll=faissRag(embedder=myembeder)

    query = "איך אני מקבל תמיכה?"

    check_embeding(ll,query)
    
    query = "מיהכן אוספים מטח?"

    check_embeding(ll,query)