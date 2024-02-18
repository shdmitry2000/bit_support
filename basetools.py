import os

from dotenv import load_dotenv
from abc import ABCMeta, abstractmethod



from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI


# from llama_index import ListIndex
# from llama_index.langchain_helpers.memory_wrapper import GPTIndexChatMemory
import logging
import sys
from langchain.agents import initialize_agent

# https://github.com/sudarshan-koirala/youtube-stuffs/blob/main/langchain/cache_llm_calls.ipynb
load_dotenv()
import basetools

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
    @staticmethod
    def getDefaultChatgpt():
        return OpenAI(temperature=0.3,max_tokens=1500 )
    
    @staticmethod  
    def getDefaultChatgptChat():
        return ChatOpenAI(temperature=0,model="gpt-4-turbo-preview",max_tokens=4000)


    