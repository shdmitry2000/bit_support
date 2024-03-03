import streamlit as st


import functools, operator, requests, os, json
from bs4 import BeautifulSoup
# from duckduckgo_search import DDGS
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict, Union
import StockTool
from vectorDbbase import baseVebtorDb
from vector_db_tools import *
from vector_dbs import *
import pandas as pd
from utility import syncdecorator
from langchain_core.runnables.config import (
    RunnableConfig,
)

from excelutility import load_data



# vectors.add_data()
# {'question': "My name's bob. How are you?", 'chat_history': [HumanMessage(content="My name's bob. How are you?", additional_kwargs={}, example=False), AIMessage(content="I'm doing well, thank you. How can I assist you today, Bob?", additional_kwargs={}, example=False)], 'answer': "I'm doing well, thank you. How can I assist you today, Bob?"}

    
class Conversational:
    
    vectors=None
    myembeder=Embeddings.getdefaultEmbading()
    

    # @syncdecorator
    # async def load_db_data(self):
    #     return await load_data()
    def getVector(self):
    

        return VectorDatabaseDAO(self.myembeder).getVector()
        


    def __init__(self) -> None:
       
            # Initialize model
        llm = ChatOpenAI(model="gpt-4-turbo-preview",temperature=0.7)

        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))



        # # 1. Define custom tools
        #-------------bit tools ------------------
        bit_tool_description="Useful for founding answers for  questions about bit application"
        bit_tool_name="bit_FAQ_search"

        # lla=vector_dbs.llamaRag(embedder=vector_dbs.Embeddings.getdefaultEmbading())
        # lla.load()
        retriver=self.getVector().getLangChainRetriver()
        la=RagTools(retriver=retriver)
        tool_bit_searcher =la.getLangchainToolDefinition(name=bit_tool_name,description=bit_tool_description)
        toolsBit= [ tool_bit_searcher ]
        #----------stock tools ------------------------------------
        tool_stock_findef=StockTool.StockTool().getToolDefinition()
        toolStock= [ tool_stock_findef ]

        #------------web content tools-------------------
        @tool("process_content", return_direct=False)
        def process_content(url: str) -> str:
            """Processes content from a webpage."""
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup.get_text()

        toolsWeb_contextParce = [ process_content]
        #-------------------------------------------- 
        
        # 2. Agents 
        # Helper function for creating agents
        def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
                # MessagesPlaceholder(variable_name="chat_history"),  # Added chat history placeholder
            ])
            agent = create_openai_tools_agent(llm, tools, prompt)
            executor = AgentExecutor(agent=agent, tools=tools)
            return executor

        # Define agent nodes
        def agent_node(state, agent, name):
            result = agent.invoke(state)
            return {"messages": [HumanMessage(content=result["output"], name=name)]}



        #  ---------------------------supervisor_chain----------------------------------
        # Create Agent Supervisor
        # members = ["Insight_Researcher","Bit_Support_Searcher" ,"Stock_Helper"]
        members = ["Bit_Support_Searcher" ]

        system_prompt_superviser = (
                """
            You are the supervisor overseeing a dynamic team of specialized workers, each with unique capabilities, identified as: {members}. Your role is to manage and direct the flow of a conversation based on a user's request. Evaluate the request, decide which worker is best suited to address it, and assign the task to them. Here’s how to proceed:
    
            Evaluate the User Request: Carefully read the user's request. Consider the skills and expertise of your team members when deciding who is best equipped to handle it.
            
            Assign the Task: Respond with the name of the worker who should act next, based on the nature of the user's request and the worker's specialization.
            
            Monitor Progress: Each worker will complete their task and update you with their results and status. If the task requires input from another worker, repeat the evaluation and assignment process.
            
            Conclude the Conversation:
            
            If all relevant queries are addressed and no further action is required, conclude the interaction with "FINISH".
            In scenarios where the user inquires about services or information outside our scope (such as PAYBOX, other apps, or banks unrelated to Bit), use the specific closing statement: "FINISH: return   לא נמצאה תושבה מתאימה לשאלה זו"
            Your goal is to ensure efficient and accurate handling of the user's request, leveraging the collective expertise of your team.
            """)

        #
        # system_prompt_superviser = (
        #     """You are a supervisor tasked with managing a conversation between the
        #      following workers:  {members}. Given the following user request,
        #      respond with the worker to act next. Each worker will perform a
        #      task and respond with their results and status.
        #      and When finished,respond with FINISH.
        #
        #      If you asked on PAYBOX or other apps or other banks and not bit,  follow final answer text:
        #      FINISH:that you are a chatbot of the Bit app and do not know how to answer these topics."""
        # )


        # system_prompt_superviser = (
        #         "You are a supervisor tasked with managing a conversation between the"
        #         " following workers:  {members}. Given the following user request,"
        #         " respond with the worker to act next. Each worker will perform a"
        #         " task and respond with their results and status. "
        #         "When finished,"
        #         " respond with FINISH."
        #     )
        options = ["FINISH"] + members
        function_def = {
            "name": "route",
            "description": "Select the next role.",
            "parameters": {
                "title": "routeSchema",
                "type": "object",
                "properties": {"next": {"title": "Next", "anyOf": [{"enum": options}] }},
                "required": ["next"],
            },
        }

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_superviser),
            MessagesPlaceholder(variable_name="messages"),
            ("system", "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options} . "),
        ]).partial(options=str(options), members=", ".join(members))

        supervisor_chain = (prompt | llm.bind_functions(functions=[function_def], function_call="route") | JsonOutputFunctionsParser())

        #------------------------------faq searcher----------------------------------------------


        # Bit_Support_Searcher_agent = create_agent(llm, toolsBit,
        #         """You are a support question answer searcher for bank application support.
        #         Based on the provided content first identify the list of topics, and what question represent every topic,
        #         then use bit_FAQ_search tool to find an answer for every question for each topic one by one.
        #         if you found answers be precision with answer and hold all data in answers include phone numbers , hours of work and amounts.
        #         if you don't found answer on any topic give final answer:
        #          will be that no data found. Be precign and polite as bank agent.
        #         """)
        Bit_Support_Searcher_agent = create_agent(llm, toolsBit,
                      """You are a support question answer searcher for bank application support. 
                      Based on the provided content first identify the list of topics, and what question represent every topic,
                      then use bit_FAQ_search tool to find an answer for every question for each topic one by one.
                      if you found answers be precision with answer and hold all data in answers include phone numbers , hours of work and amounts.
                      if you don't found answer on any topic give final answer:
                       ניתן לפנות אלינו בטלפון 6428*
                    א'-ה' - 9:00-17:00 | ו' וערבי חג - 8:30-13:00      
                          """)
        Bit_Support_Searcher_node = functools.partial(agent_node, agent=Bit_Support_Searcher_agent, name="Bit_Support_Searcher")

        #-------------------------------analist node--------------------------------------------------
        # Insight_Researcher_agent = create_agent(llm, toolsWeb_contextParce , 
        #         """You are a Insight Researcher. Do step by step. 
        #         Based on the provided content first identify the list of topics, and what question represent every topic,
        #         then ask those questions  one by one
        #         and finally find insights for each topic one by one.
        #         Include the insights and sources in the final response
        #         """)
        # Insight_Researcher_node = functools.partial(agent_node, agent=Insight_Researcher_agent, name="Insight_Researcher")


        #-------------------------stock helper (just becouse ve can) ---------------------------------------
        Stock_Helper_agent = create_agent(llm, toolStock, system_prompt="You are a responsable for get stock ticker data. you shold recive symvol  of ticker and return stock ticker price. ")
        Stock_Helper_node = functools.partial(agent_node, agent=Stock_Helper_agent, name="Stock_Helper")


        #--------------------------------all graphs and states------------------
        # Define the Agent State, Edges and Graph
        class AgentState(TypedDict):
            messages: Annotated[Sequence[BaseMessage], operator.add]
            next: str

        workflow = StateGraph(AgentState)
        workflow.add_node("Bit_Support_Searcher", Bit_Support_Searcher_node)
       # workflow.add_node("Stock_Helper", Stock_Helper_node)
        # workflow.add_node("Insight_Researcher", Insight_Researcher_node)
        workflow.add_node("supervisor", supervisor_chain)

        # Define edges
        for member in members:
            workflow.add_edge(member, "supervisor")

        conditional_map = {k: k for k in members}
        conditional_map["FINISH"] = END
        workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
        workflow.set_entry_point("supervisor")

        self.graph = workflow.compile()

        # -----------------------------end of definition----------------------


    def symantec_search(self,search_str:str):
        return self.getVector().data_search(search_str)
    
    def query_symantec_search_with_score(self,search_str:str):
        return self.getVector().query(search_str)
    
    def run_graph(self,input_message:str):
        # print("input_message",input_message)
        # history_trank=history
        response = self.graph.invoke({
            "messages": [HumanMessage(content=input_message)]
        })
        # return json.dumps(response['messages'][1].content, indent=2)
        print(response)
        try:
            return response['messages'][1].content
        except IndexError:
            return "לא נמצא תשובה!"



    def run_graphWithHistory(self,input_message:str,history=[]):
        # print("input_message",input_message)
        history_trank=history
        response = self.graph.invoke({
            "messages": [HumanMessage(content=input_message)],
            "chat_history":history_trank
        })
        # return json.dumps(response['messages'][1].content, indent=2)
        print(response)
        return response['messages'][1].content


    def invoke(
        self,
        input: Union[dict[str, Any], Any],
        config: Optional[RunnableConfig] = None,
        *,
        output_keys: Optional[Union[str, Sequence[str]]] = None,
        input_keys: Optional[Union[str, Sequence[str]]] = None,
        **kwargs: Any):
        
        if isinstance(input,str):
            response = self.run_graph(input)
        else:
            response =self.graph.invoke(input)
        return response
            
    

    def __call__(self, context):
        # Extract the 'question' from the context dictionary
        question = context['question']
        return f"{self.run_graph(question)}"


   
          
       
    
if __name__ == "__main__":   
    
    print(Conversational()(
          {
              "question":"test",
            #   "chat_history":[]
          })
    )
        
    

