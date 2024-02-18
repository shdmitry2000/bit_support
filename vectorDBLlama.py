
from enum import Enum
import json
from dotenv import load_dotenv
import llama_index
import numpy as np
import torch
from abc import abstractmethod
import os,sys
from typing import List, Optional

# from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import numpy as np
# import document

   
from enum import Enum
import json
from dotenv import load_dotenv
import llama_index
import numpy as np
import torch

import document

import abc
import os
from enum import Enum
import json
from dotenv import load_dotenv


from llama_index.core import VectorStoreIndex, SimpleDirectoryReader,SimpleKeywordTableIndex
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import StorageContext
from llama_index.core import QueryBundle

# import NodeWithScore
from llama_index.core.schema import NodeWithScore

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor


from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine

# Retrievers
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)


from llama_index.core.postprocessor import SimilarityPostprocessor

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

from typing import List, Optional

load_dotenv()

PERSIST_DIR="./indexes/lama"

import logging
import sys
# logging
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

from vectorDbbase import Embeddings,baseVebtorDb

import logging
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
        

load_dotenv()
# from rag_conversation import chain as rag_conversation_chain



    
class llamaRag(baseVebtorDb):
    
    PERCISTENT_DIRECTORY='./indexes/llama/'
    vectorstore=None
    query_engine=None
    
    
    
    def __init__(self,persist_dir=PERCISTENT_DIRECTORY,embedder=Embeddings.getdefaultEmbading()) -> None:
        
        
        self.embedder = embedder
        Settings.embed_model = embedder
        
        if os.path.exists(f"{persist_dir}/docstore.json"):
            self.load(persist_directory=persist_dir)
        else:
            os.makedirs(persist_dir, exist_ok=True)
            self.storage_context = StorageContext.from_defaults()


    @staticmethod
    def createDocument(index,title,text):
        from llama_index.core.schema import Document
        
        return  Document(id=index, title=title, text= text )
        
        
    def build_data(self,documents): 
        # Initialize LlamaIndex
        # Add data to the index

        self.vectorstore  = VectorStoreIndex.from_documents(
            documents ,storage_context=self.storage_context ,embed_model=self.embedder
        )
        
        
      


        
        # parser = SimpleNodeParser()
        # nodes = parser.get_nodes_from_documents(documents,storage_context=self.storage_context , service_context=self.service_context)
        # self.storage_context.docstore.add_documents(nodes)
        
        
        # graph_store = SimpleGraphStore()
        # self.storage_context = self.StorageContext.from_defaults(graph_store=graph_store)
        # self.storage_context.docstore.add_documents(nodes)
        
       

        

        
        # JsonDataReader = download_loader("JsonDataReader")
        # loader = JsonDataReader()
        # print(documents)
        # documents = loader.load_data(documents)
        
        # Load in a specific embedding model
        # embed_model = LangchainEmbedding(self.embedder)


# Create a service context with the custom embedding model
        

        # print("test",self.vectorstore.as_query_engine().query("test"))
        
        # parser = SimpleNodeParser()
        # nodes = parser.get_nodes_from_documents(documents)
        
        # self.storage_context.docstore.add_documents(nodes)


    def getLangChainRetriver(self):
        if self.vectorstore is None :
            raise RuntimeError("DB not found.Please load DB fierst! ")
        import llama_langchain_bridge
        return llama_langchain_bridge.LlamaIndexRetriever(index=self.vectorstore)


    def getRetriver(self):
        # if self.vectorstore is None :
        #     raise RuntimeError("DB not found.Please load DB fierst! ")
        # return llama_langchain_bridge.LlamaIndexRetriever(index=self.vectorstore,embedding=self.embedder)
       return self.vectorstore.as_retriever()
        
    
    
    def data_search_llama(self,query: str)-> str:
        
        nodes = self.getRetriver().retrieve(query)

        # filter nodes below 0.75 similarity score
        processor = SimilarityPostprocessor(similarity_cutoff=0.3)
        filtered_nodes = processor.postprocess_nodes(nodes)
        
        if (len(filtered_nodes)>0):
            replay=[]
            for node in filtered_nodes:
                answer =node.text
                # source =item.metadata
                Score =node.score
                
                replay.append((answer    , Score))
                
            return replay
        else:
            return ("No data Found!",None)
        
        
    def query(self,query: str) -> str:
        if self.query_engine is None:
            self.query_engine= self.vectorstore.as_query_engine()#(similarity_top_k=2)
        
        return self.query_engine.query(query)
    
      

    def save(self,persist_directory=PERCISTENT_DIRECTORY):
        self.vectorstore.storage_context.persist(persist_dir=persist_directory)
       
        
    # Load FAISS index from disk
    def load(self,persist_directory=PERCISTENT_DIRECTORY):       
        Settings.embed_model = self.embedder
        # rebuild storage context
        self.storage_context = StorageContext.from_defaults(persist_dir=persist_directory)
        # load index
        self.vectorstore  = load_index_from_storage(self.storage_context)





    def add_data(self,documents):
        
        if self.vectorstore is None:
            self.vectorstore  = VectorStoreIndex.from_documents(
                documents ,storage_context=self.storage_context ,embed_model=self.embedder
            )

        else:
            self.vectorstore.from_documents(documents,embed_model=self.embedder)
            
        self.vectorstore.embeddings=self.embedder
        
    
        
    def add_text(self,texts):
        index=0
        documents=[]
        for text in texts:
            index=index+1
            documents.append(llamaRag.createDocument(index,title= "title"+str(index),text=text))
            # print(documents)
        print(documents)
        self.add_data(documents)
        
    
def check_embeding(ll,query):
    # ll.load()
    from bidi.algorithm import get_display


    print(ll.__class__.__name__,"answer for:\n",get_display(str(query)),"\n",
        get_display(str(ll.data_search(query))))
        # get_display(str(ll.query(query))))
        # get_display(str(ll.data_search_llama(query))))
        
def loadDB():
    ll=llamaRag(embedder=myembeder)
    ll.delete()
    index=0
    documents=[]
    for item in document.faqs: 
        index=index+1
        document_tmp =llamaRag.createDocument(index=index,title=item['question'],text= " question : "+ str( item['question']) +" \n answer : "+ str(item['answer'])+ "\n")
                    # str(question_cleaned_string) +"\n"+ str(answer_cleaned_string))
        documents.append(document_tmp)     
    ll.build_data(documents)
    ll.save()
    
if __name__ == "__main__":         

    myembeder=Embeddings.getdefaultEmbading()

    # loadDB()


    query = "איך אני מקבל תמיכה?"

    check_embeding(llamaRag(embedder=myembeder),query)
    
    query = "מיהכן אוספים מטח?"

    check_embeding(llamaRag(embedder=myembeder),query)