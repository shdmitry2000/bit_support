import logging
import sys

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import vector_dbs
from langchain.prompts import ChatPromptTemplate,PromptTemplate

from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.agents import initialize_agent
import basetools
from langchain.agents import AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from langchain.tools.retriever import create_retriever_tool
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from basetools import projectBaseTool
# from langchain.memories import Memory
from langchain.agents import Agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# from llama_index.schema import NodeWithScore, QueryBundle, QueryType


from langchain import hub
import basetools
from basetools import projectBaseTool



class RagTools(basetools.projectBaseTool):

    rag_chain =None
    
    def __init__(self,retriver,withmemory=False,debug=True) -> None:
        super().__init__(withmemory=withmemory)
        if debug:
            logging.basicConfig(stream=sys.stdout, level=logging.INFO)
            logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

        self.retriver=retriver
        
        
        
        # self.prompt = ChatPromptTemplate(
        #     input_variables=['agent_scratchpad', 'input' , 'tools'] ,
        #     partial_variables={'chat_history': ''} ,
        #     messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['agent_scratchpad', 'chat_history', 'input',  'tools'], 
        #                 template=prompt_template))]
        #     )
    
     
    
           
    def getLangchainToolDefinition(self,name,description):
            # return Tool(
            #         name = name,
            #         func=lambda q: str(self.query(q)),
            #         description=description
            #     )
            return  create_retriever_tool(
                retriever=self.retriver,
                name=name,
                description=description,
                )
            
    def getRetriverAgent(self,prompt=None,system_prompt ="",tool_name="bit_searcher",
                         tool_definition="Useful for founding answers for  questions about bit application",
                         verbose=False,return_source_documents=False) :
        if prompt is None:
            # prompt=hub.pull("rlm/rag-prompt")
            prompt = hub.pull("hwchase17/openai-tools-agent")
        
        model=self.getDefaultChatgptChat()
        tools=[self.getLangchainToolDefinition(name=tool_name,description=tool_definition)]
        from langchain.agents import AgentExecutor, create_openai_tools_agent

        agent = create_openai_tools_agent(model, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools)
        
        return (agent,agent_executor)
            
            
    def getRetriverChain(self,prompt=None,verbose=False,return_source_documents=False) :
        
        def print_if_need(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        def contextualized_question(input: dict):
            if isinstance(input, str):
                return input
            # print("input",input)
            # print(type(input))
            if input.get("chat_history"):
                return self.getRetriverChainWithContext()
            else:
                return input["question"]
            
        if prompt is None:
            prompt=hub.pull("rlm/rag-prompt")
        # print(prompt)
        model=projectBaseTool.getDefaultChatgptChat()
        
        return (
            # {"context": self.getLangChainRetriver() | format_docs , "question": RunnablePassthrough()}
            {"context": contextualized_question| self.retriver | print_if_need , "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
            )
        
            # from langchain.chains import RetrievalQA
            # self.rag_chain  = RetrievalQA.from_llm(llm=projectBaseTool.getDefaultChatgptChat(), retriever=self.getLangChainRetriver(),
            #      prompt=prompt,
            #      verbose=verbose,return_source_documents=return_source_documents)
            


    def getRetriverChainWithContext(self,verbose=False,return_source_documents=False):
        
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        return self.getRetriverChain(prompt=contextualize_q_prompt,verbose=verbose,return_source_documents=return_source_documents)

    def getRetriverChainWithHistory(self,verbose=False,return_source_documents=False):
        
        qa_system_prompt = """You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \
        Use three sentences maximum and keep the answer concise.\

        {context}"""
        
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
                
        
        return self.getRetriverChain(prompt=qa_prompt,verbose=verbose,return_source_documents=return_source_documents)

        
        
        # return  AgentExecutor(agent=self.getRAGChain(), tools=[self.getToolDefinition()], verbose=True,handle_parsing_errors=handle_parsing_errors)

def check_RetriverChain1(ll,query):
    from bidi.algorithm import get_display


    print(ll.__class__.__name__,"answer for:\n",get_display(str(query)),"\n",
            get_display(str(ll.getRetriverChain().invoke(query))))
    
def check_agent(ll,query):
    from bidi.algorithm import get_display

    (agent,executer)=ll.getRetriverAgent()
    print(ll.__class__.__name__,"answer for:\n",get_display(str(query)),"\n",
            get_display(str(executer.invoke({"input": query}))))
    


def check_RetriverChain(ll,query,history=[]):
    # ll.load()
    from bidi.algorithm import get_display


    print(ll.__class__.__name__,"answer for:\n",get_display(str(query)),"\n",
            get_display(str(ll.getRetriverChain().invoke({
        "question": query,
        "chat_history": history,
    }))))
    
     
if __name__ == "__main__":  
    
    
    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    
    

    # print(qa.query(query))
    lla=vector_dbs.llamaRag(embedder=vector_dbs.Embeddings.getdefaultEmbading())
    # lla.load()
    retriver=lla.getLangChainRetriver()
    la=llamaRagTools(retriver=retriver)
    
    
    
    query="מה זה ביט?"
    descupdate="""
                Useful for support all questions about bit application of bank hapoalim. 
                In case we have answer the answer shoud be accepted as right answer and refrased as banking indastry answer.
                in case answer not returned- go ahead and try other posabilities to answer.
                """   
                
    
    query = "איך אני מקבל תמיכה?"

    # check_embeding1(la,query)
    # check_embeding2(la,query)
    check_RetriverChain(la,query)
    
    # question="מה זה ביט?"
    # executer= la.getExecuter()
    
    # print("answer of executer:", executer.invoke({'input':question,'chat_history':[]}))
     
      
        