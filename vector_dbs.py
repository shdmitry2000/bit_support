from enum import Enum
import json
from dotenv import load_dotenv
import llama_index
import numpy as np
import torch

import document

# export LANGCHAIN_TRACING_V2=false
# export LANGCHAIN_API_KEY='<your-api-key>'
# export LANGCHAIN_PROJECT="test_prg"
from llama_index.core.base_query_engine import BaseQueryEngine
from langchain_community.embeddings import *
# from langchain_openai import OpenAIEmbeddings
from abc import abstractmethod
import os,sys
# from config import Openai_api_key
# from langchain.llms import OpenAI
# from langchain_community.llms import OpenAI
from langchain_community.llms import OpenAI

# from langchain_openai import Claude

# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_openai import OpenAIEmbeddings 
from langchain_openai import OpenAIEmbeddings
import numpy as np
import pandas as pd
import streamlit as st
# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import redis
from langchain_core.exceptions import OutputParserException

from langchain_community.document_loaders import DirectoryLoader ,TextLoader
from llama_index.embeddings import TextEmbeddingsInference
from llama_index import ServiceContext, StorageContext, VectorStoreIndex, SimpleDirectoryReader, download_loader, load_indices_from_storage
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SimilarityPostprocessor
from langchain import hub

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
# from langchain_community.embeddings import TextEmbeddingsInference

# from langchain.chains import RAGChain
# from langchain.llms import HuggingFacePipeline
# from langchain.retrievers import ChromaRetriever
# from langchain.chains import RAGChain
# from langchain.llms import HuggingFacePipeline
# from langchain.llms import LocalLLM
# from transformers import pipeline
# from langchain.llms import HuggingFacePipeline
# from langchain import LangChain

from llama_index import VectorStoreIndex, SimpleDirectoryReader

# from llama_index import LlamaIndex,StorageContext, load_index_from_storage
from llama_index.embeddings import TextEmbeddingsInference
from llama_index.embeddings import TogetherEmbedding
from langchain_community.vectorstores.redis import Redis

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever

from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel
from langchain_community.embeddings import HuggingFaceHubEmbeddings

from llama_index.indices.base import BaseIndex

from langchain.tools.retriever import create_retriever_tool
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
# from langchain.memories import Memory
from langchain.agents import Agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from llama_index.schema import NodeWithScore, QueryBundle, QueryType

import logging
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
        

load_dotenv()
# from rag_conversation import chain as rag_conversation_chain


# Use TextEmbeddingInference to convert text to embeddings

# Define an Enum for the different encoding types
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
    
    @staticmethod    
    def getTextEmbeddingsInference():
        return TextEmbeddingsInference(
                    model_name="BAAI/bge-m3",  # required for formatting inference text,
                    timeout=60,  # timeout in seconds
                    embed_batch_size=10,  # batch size for embedding
                )
    @staticmethod        
    def getTogetherEmbedding():
        return TogetherEmbedding(
            model_name="togethercomputer/m2-bert-80M-8k-retrieval"
            #api_key="b56c0552baef29b8c6304885d2f530c75936597fd48a0f29ce0a778dd46f25b0"
        )
    
    @staticmethod
    def getHuggingFaceEmbeddings():
        return HuggingFaceEmbeddings()
        
    @staticmethod
    def getHuggingFaceMpnetbasev2Embeddings():
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return hf
        
    
    @staticmethod
    def getHuggingFaceDictaBertEmbeddings():
        model_name = "dicta-il/dictabert-heq"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return hf
    
        # # Specify the model name
        # model_name = "dicta-il/dictabert-seg"
        
        # # Load the tokenizer and the model
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        # model = AutoModel.from_pretrained(model_name)
        
        # # Tokenize the input text
        # inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        # # Generate embeddings
        # with torch.no_grad():
        #     outputs = model(**inputs)
        
        # # Get the last hidden state from the model's output
        # embeddings = outputs.last_hidden_state
        
        # return embeddings


    # @staticmethod
    # def getHuggingFaceDictaBertEmbeddings():
    #     model_name = "dicta-il/dictabert"
    #     model_kwargs = {'device': 'cpu','trust_remote_code': 'True'}
    #     encode_kwargs = {'normalize_embeddings': False}
    #     tokenizer_name="dicta-il/dictabert-joint"
        
 

    #     hf = HuggingFaceEmbeddings(
    #         model_name=model_name,
    #         model_kwargs=model_kwargs,
    #         encode_kwargs=encode_kwargs,
    #         # tokenizer_name=tokenizer_name
    #     )
        
    #     return hf



 
# class FaisseRetriever:
#     def __init__(self, faiss_index):
#         self.faiss_index = faiss_index

#     def search(self, query_embedding, k=1):
#         distances, indices = self.faiss_index.search(query_embedding, k)
#         return indices[0], distances[0] 

from typing import Any, Dict, List, cast
from llama_index.embeddings import LangchainEmbedding
        
class LlamaIndexRetriever(BaseRetriever):
    """`LlamaIndex` retriever.

    It is used for the question-answering with sources over
    an LlamaIndex data structure."""

    index: Any
    """LlamaIndex index to query."""
    query_kwargs: Dict = Field(default_factory=dict)
    """Keyword arguments to pass to the query method."""

    embedding: Any
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant for a query."""
        try:
            from llama_index.indices.base import BaseGPTIndex
            from llama_index.indices.base import BaseIndex
            from llama_index.response.schema import Response
            from llama_index.indices.vector_store.base import VectorStoreIndex
        except ImportError:
            raise ImportError(
                "You need to install `pip install llama-index` to use this retriever."
            )
        index = cast(VectorStoreIndex, self.index)
        embedding = cast(LangchainEmbedding,self.embedding)
        # print(index,self.index)
        query_engine = self.index[0].as_query_engine(
            response_mode="no_text", **self.query_kwargs
        )
        from llama_index.schema import NodeWithScore, QueryBundle, QueryType
        # print("embedding :",self.embedding)
        # custom_embedding_strs=embedding.aembed_query(query)
        if self.embedding is None:
            query_bundle = QueryBundle(query_str=query)
        else:
            query_bundle = QueryBundle(query_str=query,embedding=self.embedding.get_query_embedding(query))
        response = query_engine.query(query_bundle)
        # print(response)
        response = cast(Response, response)
        # parse source nodes
        docs = []
        from llama_index.schema import NodeRelationship
        
        for source_node in response.source_nodes:
            # metadata = source_node.extra_info or {}
            docs.append(
                # Document(page_content=source_node.source_text, metadata=metadata)
                Document(
                    page_content=source_node.node.text,
                    metadata={
                        "source": source_node.node.relationships[
                            NodeRelationship.SOURCE
                        ].node_id
                    },
                )
            )
        return docs



# class cromaRag():
#     rag_chain=None

    
#     def __init__(self,persist_dir="./indexes/chroma/") -> None:
#         # self.collection = chromadb.Collection()
#         self.persist_dir=persist_dir
#         self.chroma_client = chromadb.PersistentClient(path=self.persist_dir) 
#         self.embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
#         self.retriever = ChromaRetriever(self.chroma_client)
#         self.lc = LangChain()
#         self.llm = HuggingFacePipeline(pipeline=pipeline('text-generation'))
        
     
        
#     def encode_faqs(self,faqs):
#         encoded_faqs = []
#         for item in faqs:
#             q_embedding = self.embedder.encode([item['question']])
#             a_embedding = self.embedder.encode([item['answer']])
#             encoded_faqs.append((q_embedding, a_embedding))
#         return encoded_faqs    

#     def build_data(self):
#         encoded_faqs = self.encode_faqs(document.faqs)

#         for q_embedding, a_embedding in encoded_faqs:
#             self.chroma_client.insert(q_embedding, a_embedding)


#     def retrive(self,query):
#         if self.rag_chain is None:
#             self.retriever = ChromaRetriever(self.chroma_client)
#             self.rag_chain = RAGChain(self.retriever, self.llm)

#         response = self.rag_chain({"question": query})
#         return (response['answer']) 
    
#     def save(self):
#         pass

#     # Load FAISS index from disk
#     def load(self,path="./indexes/chroma/"):
#         self.persist_dir=path
#         self.chroma_client = chromadb.PersistentClient(path=self.persist_dir) 
        

class faissRag():
    rag_chain=None
    
    def __init__(self, embedder=Embeddings.getdefaultEmbading()) -> None:
        self.embedder=embedder
        
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        self.prompt = ChatPromptTemplate.from_template(template)

        self.model = ChatOpenAI()
        
       
        
    def getRetriver(self):
        
        # return  self.faiss_vector_store.as_retriever() 
        # print(self.faiss_vector_store.embedding_function.model)
        return  self.faiss_vector_store.as_retriever(search_type="similarity_score_threshold", 
                                      search_kwargs={"k": 2,"score_threshold": 0.7})
    
    def getToolDefinition(self,name="Get bit information",description="Useful for founding answers for question about bit application search"):
        # return Tool(
        #         name = name,
        #         func=lambda q: str(self.query(q)),
        #         description=description
        #     )
        return  create_retriever_tool(
            retriever=self.getRetriver(),
            name=name,
            description=description,
            )
          
    def query(self,question):
        # Get the retriever configured to return source documents
        retriever = self.getRetriver()
        
        docs = retriever.get_relevant_documents(question)
        # print(docs)
        # Assuming the first document contains the answer and source info
        replay=[]
        if (len(docs)>0):
            for item in docs:
                answer =item.page_content
                source =item.metadata
                replay.append((answer    , source))
                
            return replay
        else:
            return ("No data Found!",None)
        

    def build_data(self,documents):
        
        self.faiss_vector_store = FAISS.from_documents(documents,self.embedder)
        # self.faiss_vector_store.embeddings=self.embedder
        
    
               
    def save(self,path="./indexes/faiss/"):
        self.faiss_vector_store.save_local(path)
        
    # Load FAISS index from disk
    def load(self,path="./indexes/faiss/"):
        # Load the vector store from the same directory
        self.faiss_vector_store = FAISS.load_local(path, embeddings=self.embedder)
        # self.faiss_vector_store.embeddings=self.embedder

    def delete(self,path="./indexes/faiss/"):
        import shutil
        if os.path.exists(path):
            # Use shutil.rmtree() to delete the folder and all its contents
            shutil.rmtree(path)
            
    def data_search(self,query: str)-> str:
        # Get the retriever configured to return source documents
        retriever = self.getRetriver()
        
        docs = retriever.get_relevant_documents(query,embedding=self.embedder) #,metadatas=metadatas_list
        # print(docs)
        # Assuming the first document contains the answer and source info
        replay=[]
        if (len(docs)>0):
            for item in docs:
                answer =item.page_content
                source =item.metadata
                replay.append((answer    , source))
                
            return replay
        else:
            return ("No data Found!",None)
        

    # def retrive(self,query):
    #     if self.rag_chain is None:
    #         # self.rag_chain = (
    #         # {"context": self.getRetriver(), "question": RunnablePassthrough()}
    #         # | self.prompt
    #         # | self.model
    #         # | StrOutputParser()
    #         # )
            
    #         from langchain.chains import RetrievalQA
    #         self.rag_chain  = RetrievalQA.from_llm(llm=OpenAI(), retriever=self.getRetriver(),
    #              prompt=self.prompt,
    #              verbose=True,return_source_documents=True)
        
            
            
        return self.rag_chain.invoke(query)

class RedisRag():
    rag_chain=None
   
    rag_chain =None
    redis_schema="redis_schema.yaml"
    
    
    def __init__(self, redis_url="redis://localhost:6379", index_name="bit" ,embedder=Embeddings.getdefaultEmbading()) -> None:
        self.embedder=embedder
        self.redis_url = redis_url
        self.index_name = index_name
        self.embedder = embedder
        # self.rds = redis.Redis.from_url(self.redis_url)

        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        self.prompt = ChatPromptTemplate.from_template(template)

        self.model = ChatOpenAI()
        
        

        
        
    def getRetriver(self):
        
        # return self.rds.as_retriever()
        return  self.rds.as_retriever(search_type="similarity_score_threshold", 
                                      search_kwargs={"k": 2,"score_threshold": 0.7})
    

    
    
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

    def query(self,question):
        # Get the retriever configured to return source documents
        retriever = self.getRetriver()
        
        docs = retriever.get_relevant_documents(question)
        # print(docs)
        # Assuming the first document contains the answer and source info
        replay=[]
        if (len(docs)>0):
            for item in docs:
                answer =item.page_content
                source =item.metadata
                replay.append((answer    , source))
                
            return replay
        else:
            return ("No data Found!",None)
    
    

    def build_data(self,documents):
        self.rds = redis.Redis.from_documents(
            documents,
            self.embedder,
            redis_url=self.redis_url,
            index_name=self.index_name,
        )
        
        self.rds.write_schema(self.redis_schema)
        
              
    def save(self,path="./indexes/redis/"):
        pass
        
    # Load FAISS index from disk
    def load(self,path="./indexes/redis/"):
        self.rds = Redis.from_existing_index(
        self.embedder,
        index_name=self.index_name,
        redis_url=self.redis_url,
        schema=self.redis_schema,
        )

    def delete(self):
        Redis.drop_index(
            index_name=self.index_name, delete_documents=True, redis_url=self.redis_url )

    def getToolDefinition(self,name="Get bit information",description="Useful for founding answers for question about bit application search"):
        # return Tool(
        #         name = name,
        #         func=lambda q: str(self.query(q)),
        #         description=description
        #     )
        return  create_retriever_tool(
            retriever=self.getRetriver(),
            name=name,
            description=description,
            )
        
    # def query(self,question):
    #     # Get the retriever configured to return source documents
    #     retriever = self.getRetriver()
        
    #     docs = retriever.get_relevant_documents(question)
        
    #     # Assuming the first document contains the answer and source info
    #     if (len(docs)>0):
    #         answer = docs[0].page_content
    #         source = docs[0].metadata
    #     else:
    #         answer="No data Found!"
    #         source=None
        
    #     return  (answer    , source)
    
    def data_search(self,query: str)-> str:
        # Get the retriever configured to return source documents
        retriever = self.getRetriver()
        
        docs = retriever.get_relevant_documents(query,embedding=self.embedder) #,metadatas=metadatas_list
        # print(docs)
        # Assuming the first document contains the answer and source info
        replay=[]
        if (len(docs)>0):
            for item in docs:
                answer =item.page_content
                source =item.metadata
                replay.append((answer    , source))
                
            return replay
        else:
            return ("No data Found!",None)

    # def retrive(self,query):
    #     if self.rag_chain is None:
    #         # self.rag_chain = (
    #         # {"context": self.getRetriver(), "question": RunnablePassthrough()}
    #         # | self.prompt
    #         # | self.model
    #         # | StrOutputParser()
    #         # )
            
    #         from langchain.chains import RetrievalQA
    #         self.rag_chain  = RetrievalQA.from_llm(llm=OpenAI(), retriever=self.getRetriver(),
    #              prompt=self.prompt,
    #              verbose=True,return_source_documents=True)
        
    #     return self.rag_chain.invoke(query)
    
    # def retrive(self,query):
        
   
      

class llamaRag():
    query_engine=None
    rag_chain=None
    PERSIST_DIR='./indexes/llama/'

    
    def __init__(self,persist_dir=PERSIST_DIR,embedder=Embeddings.getdefaultEmbading()) -> None:
        
        
        from llama_index.embeddings import LangchainEmbedding
        self.embedder = LangchainEmbedding(
            embedder
        )
        
        self.service_context = ServiceContext.from_defaults(chunk_size=1000,embed_model=self.embedder)
        
        
        if os.path.exists(f"{persist_dir}/docstore.json"):
            
            self.load(persist_dir=persist_dir)
        else:
            os.makedirs(persist_dir, exist_ok=True)
            self.storage_context = StorageContext.from_defaults()

        
        # template = """Answer the question based only on the following context:
        # {context}

        # Question: {question}
        # """
        # self.prompt = ChatPromptTemplate.from_template(template)

        # self.model = ChatOpenAI()

    @staticmethod
    def createDocument(index,title,text):
        from llama_index.schema import Document
        
        return  Document(id=index, title=title, text= text )
        
    
    def build_data(self,documents): 
        # Initialize LlamaIndex
        # Add data to the index

        from llama_index import download_loader, VectorStoreIndex, StorageContext, ServiceContext 
        
        self.index  = VectorStoreIndex.from_documents(
            documents ,storage_context=self.storage_context , service_context=self.service_context
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
        

        # print("test",self.index.as_query_engine().query("test"))
        
        # parser = SimpleNodeParser()
        # nodes = parser.get_nodes_from_documents(documents)
        
        # self.storage_context.docstore.add_documents(nodes)



    def getLangchainRetriver(self):
    #    from langchain.retrievers.llama_index import LlamaIndexRetriever
        if self.index is None :
            raise RuntimeError("DB not found.Please load DB fierst! ")
        return LlamaIndexRetriever(index=self.index,embedding=self.embedder)
    #    retriever = VectorIndexRetriever(
    #         index=self.index,
    #         similarity_top_k=2,
    #     )
    
    
    def query(self,query: str) -> str:
        if self.query_engine is None:
            self.query_engine= self.index[0].as_query_engine()
        
        
        # custom_embedding_strs=embedding.aembed_query(query)
        if self.embedder is None:
            query_bundle = QueryBundle(query_str=query)
        else:
            query_bundle = QueryBundle(query_str=query,embedding=self.embedder.get_query_embedding(query))
        
        
        return self.query_engine.query(query_bundle)

    def data_search(self,query: str)-> str:
        # Get the retriever configured to return source documents
        retriever = self.getLangchainRetriver()
        
        docs = retriever.get_relevant_documents(query,embedding=self.embedder) #,metadatas=metadatas_list
        # print(docs)
        # Assuming the first document contains the answer and source info
        replay=[]
        if (len(docs)>0):
            for item in docs:
                answer =item.page_content
                source =item.metadata
                replay.append((answer    , source))
                
            return replay
        else:
            return ("No data Found!",None)
    
    # def getLangchainToolDefinition(self,name="bit_FAQ_search",description="Useful for founding answers for Frequently asked questions about bit application"):
    #     # return Tool(
    #     #         name = name,
    #     #         func=lambda q: str(self.query(q)),
    #     #         description=description
    #     #     )
    #     return  create_retriever_tool(
    #         retriever=self.getLangchainRetriver(),
    #         name=name,
    #         description=description,
    #         )
    # def getRetriverAgent(self,prompt=None,verbose=False,return_source_documents=False) :
    #     if prompt is None:
    #         # prompt=hub.pull("rlm/rag-prompt")
    #         prompt = hub.pull("hwchase17/openai-tools-agent")
        
    #     model=projectBaseTool.getDefaultChatgptChat()
    #     tools=[self.getLangchainToolDefinition()]
    #     from langchain.agents import AgentExecutor, create_openai_tools_agent

    #     agent = create_openai_tools_agent(model, tools, prompt)
    #     agent_executor = AgentExecutor(agent=agent, tools=tools)
        
    #     return (agent,agent_executor)
            
            
    # def getRetriverChain(self,prompt=None,verbose=False,return_source_documents=False) :
        
    #     def print_if_need(docs):
    #         return "\n\n".join(doc.page_content for doc in docs)
        
    #     def contextualized_question(input: dict):
    #         if isinstance(input, str):
    #             return input
    #         # print("input",input)
    #         # print(type(input))
    #         if input.get("chat_history"):
    #             return self.getRetriverChainWithContext()
    #         else:
    #             return input["question"]
         
    #     if prompt is None:
    #         prompt=hub.pull("rlm/rag-prompt")
    #     # print(prompt)
    #     model=projectBaseTool.getDefaultChatgptChat()
        
    #     return (
    #         # {"context": self.getLangchainRetriver() | format_docs , "question": RunnablePassthrough()}
    #         {"context": contextualized_question| self.getLangchainRetriver() | print_if_need , "question": RunnablePassthrough()}
    #         | prompt
    #         | model
    #         | StrOutputParser()
    #         )
        
    #      # from langchain.chains import RetrievalQA
    #         # self.rag_chain  = RetrievalQA.from_llm(llm=projectBaseTool.getDefaultChatgptChat(), retriever=self.getLangchainRetriver(),
    #         #      prompt=prompt,
    #         #      verbose=verbose,return_source_documents=return_source_documents)
           
    
    
    # def getRetriverChainWithContext(self,verbose=False,return_source_documents=False):
        
    #     contextualize_q_system_prompt = """Given a chat history and the latest user question \
    #     which might reference context in the chat history, formulate a standalone question \
    #     which can be understood without the chat history. Do NOT answer the question, \
    #     just reformulate it if needed and otherwise return it as is."""
        
    #     contextualize_q_prompt = ChatPromptTemplate.from_messages(
    #         [
    #             ("system", contextualize_q_system_prompt),
    #             MessagesPlaceholder(variable_name="chat_history"),
    #             ("human", "{question}"),
    #         ]
    #     )
    #     return self.getRetriverChain(prompt=contextualize_q_prompt,verbose=verbose,return_source_documents=return_source_documents)

    # def getRetriverChainWithHistory(self,verbose=False,return_source_documents=False):
        
    #     qa_system_prompt = """You are an assistant for question-answering tasks. \
    #     Use the following pieces of retrieved context to answer the question. \
    #     If you don't know the answer, just say that you don't know. \
    #     Use three sentences maximum and keep the answer concise.\

    #     {context}"""
        
    #     qa_prompt = ChatPromptTemplate.from_messages(
    #         [
    #             ("system", qa_system_prompt),
    #             MessagesPlaceholder(variable_name="chat_history"),
    #             ("human", "{question}"),
    #         ]
    #     )
                
        
    #     return self.getRetriverChain(prompt=qa_prompt,verbose=verbose,return_source_documents=return_source_documents)
    

    
     # Save Llama index to disk
    def save(self, persist_dir="./indexes/llama/"):
        self.index.storage_context.persist(persist_dir=persist_dir)

    # Load Llama index from disk
    def load(self,persist_dir="./indexes/llama/"):
        self.storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        self.index = load_indices_from_storage(self.storage_context)
 
    def delete(self,path="./indexes/llama/"):
        import shutil
        if os.path.exists(path):
            # Use shutil.rmtree() to delete the folder and all its contents
            shutil.rmtree(path)



def check_embeding(ll,query):
    # ll.load()
    from bidi.algorithm import get_display


    print(ll.__class__.__name__,"answer for:\n",get_display(str(query)),"\n",
        get_display(str(ll.data_search(query))))
    
# def check_embeding2(ll,query):
#     # ll.load()
#     from bidi.algorithm import get_display


#     print(ll.__class__.__name__,"answer for:\n",get_display(str(query)),"\n",
#             get_display(str(ll.getRetriverChain().invoke(query))))
    
# def check_embeding3(ll,query):
#     # ll.load()
#     from bidi.algorithm import get_display

#     (agent,executer)=ll.getRetriverAgent()
#     print(ll.__class__.__name__,"answer for:\n",get_display(str(query)),"\n",
#             get_display(str(executer.invoke({"input": query}))))
    


# def check_embeding1(ll,query,history=[]):
#     # ll.load()
#     from bidi.algorithm import get_display


#     print(ll.__class__.__name__,"answer for:\n",get_display(str(query)),"\n",
#             get_display(str(ll.getRetriverChain().invoke({
#         "question": query,
#         "chat_history": history,
#     }))))
    

    
if __name__ == "__main__":         

    myembeder=Embeddings.getdefaultEmbading()

    # ll=faissRag(embedder=myembeder)
    # ll.delete()
    # ll.build_data(documents=document.get_documents())
    # ll.save() 

    # ll=RedisRag(embedder=myembeder)
    # ll.delete()
    # ll.build_data(documents=document.get_documents())
    # ll.save()

    # ll=llamaRag(embedder=myembeder)
    # ll.delete()
    # index=0
    # documents=[]
    # for item in document.faqs: 
    #     index=index+1
    #     document_tmp =llamaRag.createDocument(index=index,title=item['question'],text= " question : "+ str( item['question']) +" \n answer : "+ str(item['answer'])+ "\n")
    #                 # str(question_cleaned_string) +"\n"+ str(answer_cleaned_string))
    #     documents.append(document_tmp)     
    # ll.build_data(documents)
    # ll.save()


    query = "איך אני מקבל תמיכה?"

    # check_embeding1(faissRag(embedder=myembeder),query)
    # check_embeding(RedisRag(embedder=myembeder),query)
    check_embeding(llamaRag(embedder=myembeder),query)

    query = "מיהכן אוספים מטח?"

    # check_embeding1(faissRag(embedder=myembeder),query)
    # check_embeding(RedisRag(embedder=myembeder),query)
    # check_embeding2(llamaRag(embedder=myembeder),query)


    