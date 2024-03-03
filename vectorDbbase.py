from abc import abstractmethod
import os
from typing import List, Optional
import uuid
from chromadb import EmbeddingFunction
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import numpy as np
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer, util
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from ast import List
from typing import Any, Dict, Optional, Union

import chromadb
import sentence_transformers

EMBEDDING_MODEL='intfloat/multilingual-e5-large'
# less good
#  "all-MiniLM-L6-v2",
#less good
# 'imvladikon/sentence-transformers-alephbert'
# slow and hung
# 'Salesforce/SFR-Embedding-Mistral'

#slow
# EMBEDDING_MODEL='Salesforce/SFR-Embedding-Mistral'
load_dotenv()


class HebSentenceTransformerEmbeddingFunction(embedding_functions.SentenceTransformerEmbeddingFunction):

    # If you have a beefier machine, try "gtr-t5-large".
    # for a full list of options: https://huggingface.co/sentence-transformers, https://www.sbert.net/docs/pretrained_models.html
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL ,
        device: str = "cpu",
        normalize_embeddings: bool = False,
    ):
        if model_name not in self.models:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ValueError(
                    "The sentence_transformers python package is not installed. Please install it with `pip install sentence_transformers`"
                )
            self.models[model_name] = SentenceTransformer(model_name, device=device,cache_folder="./encoder_cache/")
        self._model = self.models[model_name]
        self._normalize_embeddings = normalize_embeddings



class HebSentenceTransformerEmbeddings(HuggingFaceEmbeddings):

    model_name: str = EMBEDDING_MODEL
   
    # """Model name to use."""
    cache_folder: Optional[str] = "./encoder_cache/"
    
    device: str = "cpu"
    


# from chromadb import  Documents, EmbeddingFunction , Embeddings
class HebChromaEmbeddingFunction(chromadb.EmbeddingFunction[chromadb.Documents]):
    # model = SentenceTransformer('dicta-il/dictabert-seg',cache_folder="./encoder_cache/")
    model = SentenceTransformer(EMBEDDING_MODEL,cache_folder="./encoder_cache/")
    # model = SentenceTransformer('intfloat/multilingual-e5-large',cache_folder="./encoder_cache/")
    # model = SentenceTransformer(model_name_or_path='imvladikon/sentence-transformers-alephbert' ,cache_folder="./encoder_cache/")
    
    def getEmbeddingModel(self):
        return self.model
    
    def embed(self, input: Union[str, list[str]]) -> chromadb.Embeddings:
        if isinstance(input, str):
            input = [input]
        return self.getEmbeddingModel().encode(input)
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs."""
        return self.getEmbeddingModel().encode(texts)
    
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        # embed the documents somehow
        return self.embed(input)


    
class Embeddings():

    @staticmethod  
    def getdefaultEmbading():
        return Embeddings.getHebSentenceTransformerEmbeddings()
    
    @staticmethod       
    def getOpenAiEmbadings():
        return OpenAIEmbeddings(model="text-embedding-ada-002")
    
    @staticmethod       
    def getOpenAi3SmallEmbadings():
        return OpenAIEmbeddings(model="text-embedding-3-small")
    
    @staticmethod       
    def getOpenAi3LargeEmbadings():
        return OpenAIEmbeddings(model="text-embedding-3-large")
    @staticmethod  
    # def getHebEembedding():
    #     return SentenceTransformerEmbeddings(model_name="dicta-il/dictabert-seg")

    @staticmethod 
    def getHebSentenceTransformerEmbeddings():
        #return HebSentenceTransformerEmbeddings(model_name='intfloat/multilingual-e5-large',cache_folder= "./encoder_cache/")
        # return HebSentenceTransformerEmbeddings(model_name='Salesforce/SFR-Embedding-Mistral',cache_folder= "./encoder_cache/")
        return HebSentenceTransformerEmbeddings()
    @staticmethod 
    def getHebSentenceTransformerEmbeddingFunction():      
        return  HebSentenceTransformerEmbeddingFunction()
  
    
    @staticmethod 
    def getHebChromaEmbeddingFunction():
        return HebChromaEmbeddingFunction()
    
# documents=[]
# document_tmp = createDocument(metadata='question',page_content= " i am happy")
# documents.append(document_tmp) 

# db_openAIEmbedd = FAISS.from_documents(documents, Embeddings.getdefaultEmbading())
# retriever_openai = db_openAIEmbedd.as_retriever(search_kwargs={"k": 3})


    
class baseVebtorDb():
    
    PERCISTENT_DIRECTORY="./indexes/base/"
    
    def __init__(self, persist_dir,embedder=Embeddings.getdefaultEmbading()) -> None:
        self.embedder=embedder
        if os.path.exists(f"{persist_dir}"):
            self.load(persist_directory=persist_dir)
        else:
            self.vectorstore=None 
            
    
    @abstractmethod
    def getLangChainRetriver(self):
        pass
        
    
    def data_search(self,query: str):
        
        
        # Get the retriever configured to return source documents
        retriever = self.getLangChainRetriver()
        
        # self.asimilarity_search_with_score_by_vector
        docs = retriever.get_relevant_documents(query,embedding=self.embedder) #,metadatas=metadatas_list

        return docs
    
    
    @abstractmethod
    def getVector(self):
        pass
    
    def query(self,query: str):
        vector=self.getVector()
        # return vector.similarity_search(query)
        # return vector.similarity_search_with_score(query)
        return vector.similarity_search_with_relevance_scores(query)

    @abstractmethod
    def save(self,persist_directory):
        pass
        
    @abstractmethod
    def load(self,persist_directory):
        pass

    @abstractmethod
    def delete(self,path=PERCISTENT_DIRECTORY):
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
        # if self.vectorstore is None:
        #     documents = [chromaRag.createDocument(page_content=text) for text in texts]
        #     # print(documents)
        #     self.vectorstore=Chroma.from_documents(documents,self.embedder)
        # else:
        #     self.vectorstore.add_texts(texts)    
        pass
    
    @staticmethod  
    def getUUID_ID(count=1):
        return [ str(uuid.uuid1()) for i in range(count) ] 
           
           
    @abstractmethod
    def printDb(self):
        raise RuntimeError("not implemented yet")


class basevectorDBLangchain(baseVebtorDb):
    
    
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
        
        
        
        
class basevectorDBLlamaIndex(baseVebtorDb):
    
    
    
    @staticmethod
    def createDocument(index,title,text):
        from llama_index.core.schema import Document
        
        return  Document(doc_id=index, title=title, text= text )
    
    @staticmethod
    def createTextNode(text):
        from llama_index.core.schema import TextNode
        
        return  TextNode( text= text )
    
    
        
        
    # @staticmethod
    # def createDocument(index,title,text):
    #     from llama_index.core.schema import Document
        
    #     return  Document(id=index, title=title, text= text )
     