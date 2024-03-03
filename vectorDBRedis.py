import os
from typing import List
# from langchain_community.vectorstores import redis
from langchain_community.vectorstores.redis import Redis
from langchain_core.documents import Document
import numpy as np
from excelutility import getExcelDatainJson

from vectorDbbase import Embeddings,baseVebtorDb, basevectorDBLangchain




    
class redisRag(baseVebtorDb):
    
   
    redis_schema="redis_schema.yaml"
    
    
    def __init__(self, redis_url="redis://localhost:6379", index_name="bit" ,embedder=Embeddings.getdefaultEmbading()) -> None:
        self.embedder=embedder
        self.redis_url = redis_url
        self.index_name = index_name
        
        self.load()
   
        # self.rds = Redis.from_url(self.redis_url)

        

    def getLangChainRetriver(self):
        
        # return  self.faiss_vector_store.as_retriever() 
        # print(self.faiss_vector_store.embedding_function.model)
        return  self.rds.as_retriever(search_type="similarity_score_threshold", 
                                      search_kwargs={"k": 2,"score_threshold": 0.1})
    

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
            
            
          
    def getVector(self):
        return self.rds
        
    # Load FAISS index from disk
    def load(self,persist_directory="./redis"):
        self.rds = Redis.from_existing_index(
            self.embedder,
            index_name=self.index_name,
            redis_url=self.redis_url,
            schema=self.redis_schema,
        )
        
    def delete(self,path):
        Redis.drop_index(
            index_name=self.index_name, delete_documents=True, redis_url=self.redis_url )

            
    def build_data(self,documents):
        self.rds = Redis.from_documents(
            documents,
            self.embedder,
            redis_url=self.redis_url,
            index_name=self.index_name,
        )
        
        self.rds.write_schema(self.redis_schema)
        
        
    def add_data(self,documents):
        
        self.rds = Redis.from_documents(
            documents,
            self.embedder,
            redis_url=self.redis_url,
            index_name=self.index_name,
        )
        
    def add_text(self,texts):
        documents = [basevectorDBLangchain.createDocument(page_content=text) for text in texts]
        self.add_data(documents)
        
    def process_results(self, results):
        # Process the results here
        processed_results = []
        for result in results:
            # Assume each result has a 'source' field with the source document
            processed_results.append({
                'source': result['source'],
                # ... other fields
            })
        return processed_results

    


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
        
      

def loadRedisDB(faqs,myembeder):
    ll=redisRag(embedder=myembeder)
    ll.delete()
    documents=loadDocumentsfromFaq(faqs)
    ll.build_data(documents)
    ll.save() 
    
    
def loadRedisDBfromExcel(redisRag):
    json_data =  getExcelDatainJson()
    print("load additional questions to redis")
    documents=loadDocumentsfromFaq(json_data) 
    redisRag.add_data(documents)    
    # redisRag.printDb()
    return redisRag 


def getVectorDB(embeding):
    ll=redisRag(embedder=embeding)
    return loadRedisDBfromExcel(ll)  

if __name__ == "__main__":         

    myembeder=Embeddings.getdefaultEmbading()

    loadRedisDB(document.faqs,myembeder)
    

    query = "איך אני מקבל תמיכה?"

    check_embeding(redisRag(embedder=myembeder),query)
    
    query = "מיהכן אוספים מטח?"

    check_embeding(redisRag(embedder=myembeder),query)
    
    
    