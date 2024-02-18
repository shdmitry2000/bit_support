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

# Retrievers
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)

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

# class BaseSynthesizer(abc):
#     """Response builder class."""

#     def __init__(
#         self,
#         llm: Optional[LLM] = None,
#         streaming: bool = False,
#     ) -> None:
#         """Init params."""
#         self._llm = llm or Settings.llm
#         self._callback_manager = Settings.callback_manager
#         self._streaming = streaming

#     @abstractmethod
#     def get_response(
#         self,
#         query_str: str,
#         text_chunks: Sequence[str],
#         **response_kwargs: Any,
#     ) -> RESPONSE_TEXT_TYPE:
#         """Get response."""
#         ...

#     @abstractmethod
#     async def aget_response(
#         self,
#         query_str: str,
#         text_chunks: Sequence[str],
#         **response_kwargs: Any,
#     ) -> RESPONSE_TEXT_TYPE:
#         """Get response."""


def createDocument(index,title,text,embedding=None):
        from llama_index.core.schema import Document
        
        return  Document(id=index, title=title, text= text ,embedding=embedding)
    
 
 
 

 
document_tmp=  createDocument("1","title","the is a grate date , and ts 11/1")  
documents=[document_tmp]


embed_model=OpenAIEmbedding()


# global
Settings.embed_model = OpenAIEmbedding()

# per-index
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

query=" grate date"


from llama_index.core.postprocessor import SimilarityPostprocessor

nodes = index.as_retriever().retrieve(query)

# filter nodes below 0.75 similarity score
processor = SimilarityPostprocessor(similarity_cutoff=0.75)
filtered_nodes = processor.postprocess_nodes(nodes)
print(type(filtered_nodes[0]),filtered_nodes[0])
exit(0)

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=2,
)




# configure response synthesizer
response_synthesizer = get_response_synthesizer()

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer= response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
)

# query
response = query_engine.query(query)
print(response)

exit(0)


query_engine = index.as_query_engine()
print(query_engine.query(query))
exit(0)

query_engine = index.as_query_engine()
print(query_engine.query(query))
exit(0)



exit(0)





nodes = Settings.get_nodes_from_documents(documents)


# initialize storage context (by default it's in-memory)
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
keyword_index = SimpleKeywordTableIndex(nodes, storage_context=storage_context)

class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        keyword_retriever: KeywordTableSimpleRetriever,
        mode: str = "AND",
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes


from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine

# define custom retriever
vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=2)
keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index)
custom_retriever = CustomRetriever(vector_retriever, keyword_retriever)

# define response synthesizer
response_synthesizer = get_response_synthesizer()

# assemble query engine
custom_query_engine = RetrieverQueryEngine(
    retriever=custom_retriever,
    response_synthesizer=response_synthesizer,
)

# vector query engine
vector_query_engine = RetrieverQueryEngine(
    retriever=vector_retriever,
    response_synthesizer=response_synthesizer,
)
# keyword query engine
keyword_query_engine = RetrieverQueryEngine(
    retriever=keyword_retriever,
    response_synthesizer=response_synthesizer,
)

query=" grate date"
response = custom_query_engine.query(
    query
)

print(response)


exit(0)
    
# document_tmp=  createDocument("1","title","the is a grate date")  
# documents=[document_tmp]
# index = VectorStoreIndex.from_documents(documents)


documents = SimpleDirectoryReader("/Users/dmitryshlymovich/workspace/chatgpt/tests/XpayFinance_LLM/data").load_data()
print(type(documents[0]),documents)
index = VectorStoreIndex.from_documents(documents)

query="איך אני מקבל תמיכה?"
query_engine = index.as_query_engine()
query_engine.query(query)
index.storage_context.persist(persist_dir=PERSIST_DIR)

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
# load index
index = load_index_from_storage(storage_context)