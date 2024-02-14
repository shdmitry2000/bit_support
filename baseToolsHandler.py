import os

from dotenv import load_dotenv
from abc import ABCMeta, abstractmethod

import langchain
# from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import (
    ConversationalRetrievalChain,
    LLMChain
)

# from langchain import HuggingFaceHub
# from utilities import load_api_key


# from langchain.vectorstores import Chroma
# from langchain.vectorstores.faiss import FAISS
# from langchain.chat_models import ChatOpenAI , ChatAnthropic
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory,ConversationSummaryMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT,
#                                                      QA_PROMPT)
"""Create a ChatVectorDBChain for question/answering."""
# from langchain.callbacks.base import AsyncCallbackManager
from langchain.callbacks.manager import AsyncCallbackManager
# from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ChatVectorDBChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain 
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.vectorstores.base import VectorStore
from langchain.callbacks.tracers import LangChainTracer
import  openai

from transformers import TFAutoModelForQuestionAnswering
from langchain.chains import AnalyzeDocumentChain 
from langchain.chains.summarize import load_summarize_chain

from langchain.cache import InMemoryCache
# from langchain.callbacks import get_openai_callback
# We can do the same thing with a SQLite cache
from langchain.cache import SQLiteCache
# from langchain.chat_models import ChatOpenAI
# from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
# from langchain.llms import OpenAI
# from langchain import SerpAPIWrapper
# from langchain.agents.tools import Tool
# from langchain import LLMMathChain

from llama_index import ListIndex
from llama_index.langchain_helpers.memory_wrapper import GPTIndexChatMemory
import logging
import sys
# from langchain.llms import OpenAIChat
from langchain.agents import initialize_agent

# https://github.com/sudarshan-koirala/youtube-stuffs/blob/main/langchain/cache_llm_calls.ipynb
load_dotenv()

class projectBaseTool():

    @staticmethod
    def getDefMemory(name="all_chat_history"):
        index = ListIndex([])
        memory = GPTIndexChatMemory(
            index=index, 
            memory_key=name, 
            query_kwargs={"response_mode": "compact"},
            # return_source returns source nodes instead of querying index
            return_source=True,
            # return_messages returns context in message format
            return_messages=True
        )
        return memory

    def __init__(self,memory_name="chat_history",withmemory=False) -> None:
        self.withmemory=withmemory   
        if withmemory :
            
            self.memory =projectBaseTool.getDefMemory(memory_name)
        else:
            self.memory =None

            
       


    