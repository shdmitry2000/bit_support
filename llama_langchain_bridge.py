from langchain_core.documents import Document

from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever


from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_openai import OpenAIEmbeddings
from typing import Any, Dict, List, cast

from typing import Any, Dict, List, cast
# from langchain_core.documents import Document
    
    
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
            # from llama_index.indices.base import BaseGPTIndex
            # from llama_index.indices.base import BaseIndex
            # from llama_index.response.schema import Response
            # from llama_index.indices.vector_store.base import VectorStoreIndex
            from llama_index.core import VectorStoreIndex, get_response_synthesizer
            from llama_index.core.retrievers import VectorIndexRetriever
            from llama_index.core.query_engine import RetrieverQueryEngine
            from llama_index.core.postprocessor import SimilarityPostprocessor
            from llama_index.core.schema import NodeWithScore
            from llama_index.core import VectorStoreIndex, SimpleDirectoryReader,SimpleKeywordTableIndex

        except ImportError:
            raise ImportError(
                "You need to install `pip install llama-index` to use this retriever."
            )
        index = cast(VectorStoreIndex, self.index)
        # print("self.index",self.index)
        # retriver=self.index.as_retriever(**self.query_kwargs)
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=2,
        )

        nodes = retriever.retrieve(query)

        # filter nodes below 0.75 similarity score
        processor = SimilarityPostprocessor(similarity_cutoff=0.3)
        filtered_nodes = processor.postprocess_nodes(nodes)
        
        docs = []
        if (len(filtered_nodes)>0):
            
            for node in filtered_nodes:
                print("node",type(node),node)
                docs.append(
                    # Document(page_content=source_node.source_text, metadata=metadata)
                    Document(
                        page_content=node.text,
                        metadata={
                            "source": node.id_ + " Score:"+str(node.score)
                        },
                    )
                )
        return docs



    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant for a query."""
        try:
            from llama_index.core.schema import NodeRelationship
            from llama_index.core.schema import NodeWithScore
            from llama_index.core import Response
            from llama_index.core import VectorStoreIndex, get_response_synthesizer
            from llama_index.core.retrievers import VectorIndexRetriever
            from llama_index.core.query_engine import RetrieverQueryEngine
            from llama_index.core.postprocessor import SimilarityPostprocessor
            from llama_index.core.schema import NodeWithScore
            from llama_index.core import VectorStoreIndex, SimpleDirectoryReader,SimpleKeywordTableIndex

        except ImportError:
            raise ImportError(
                "You need to install `pip install llama-index` to use this retriever."
            )
        index = cast(VectorStoreIndex, self.index)
        # print("self.index",self.index)
        # embedding = cast(LangchainEmbedding,self.embedding)
        query_engine = self.index.as_query_engine(
            response_mode="no_text", **self.query_kwargs
        )
        response = query_engine.query(query)
        
        # print(response)
        response = cast(Response, response)
        # parse source nodes
        docs = []
        
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