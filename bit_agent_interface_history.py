import streamlit as st


import functools, operator, requests, os, json
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
import StockTool
from vector_db_tools import *
from vector_dbs import *

# Initialize model
llm = ChatOpenAI(model="gpt-4-turbo-preview")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))



# # 1. Define custom tools
#-------------bit tools ------------------
bit_tool_description="Useful for founding answers for  questions about bit application"
bit_tool_name="bit_FAQ_search"

lla=vector_dbs.llamaRag(embedder=vector_dbs.Embeddings.getdefaultEmbading())
# lla.load()
retriver=lla.getLangchainRetriver()
la=llamaRagTools(retriver=retriver)
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
        MessagesPlaceholder(variable_name="chat_history"),  # Added chat history placeholder
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
members = ["Insight_Researcher","Bit_Support_Searcher" ,"Stock_Helper"]

system_prompt_superviser = (
    "As a supervisor, your role is to oversee a dialogue of "
    " workers: {members}. Based on the user's request,"
    " determine which worker should take the next action in need any. Each worker is responsible for"
    " executing a specific task and reporting back their findings and progress. Once all tasks are complete,"
    " indicate with 'FINISH'.If no good answer gave by workers or no worker for task selected : give answer that you not find the answer. Be polite as banker ! "
)

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

prompt_superviser = ChatPromptTemplate.from_messages([
    ("system", system_prompt_superviser),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}"),
]).partial(options=str(options), members=", ".join(members))

supervisor_chain = (prompt_superviser | llm.bind_functions(functions=[function_def], function_call="route") | JsonOutputFunctionsParser())

#------------------------------faq searcher----------------------------------------------


Bit_Support_Searcher_agent = create_agent(llm, toolsBit, "You are a support question answer searcher.use bit_FAQ_search tool to find an answer.")
Bit_Support_Searcher_node = functools.partial(agent_node, agent=Bit_Support_Searcher_agent, name="Bit_Support_Searcher")

#-------------------------------analist node--------------------------------------------------
Insight_Researcher_agent = create_agent(llm, toolsWeb_contextParce , 
        """You are a Insight Researcher. Do step by step. 
        Based on the provided content first identify the list of topics, and what question represent every topic,
        then ask those questions  one by one
        and finally find insights for each topic one by one.
        Include the insights and sources in the final response
        """)
Insight_Researcher_node = functools.partial(agent_node, agent=Insight_Researcher_agent, name="Insight_Researcher")


#-------------------------stock helper (just becouse ve can) ---------------------------------------
Stock_Helper_agent = create_agent(llm, toolStock, 
        """You are a responsable for get stock ticker data. you shold recive symvol  of ticker 
        and return stock ticker price.
        """)
Stock_Helper_node = functools.partial(agent_node, agent=Stock_Helper_agent, name="Stock_Helper")


#--------------------------------all graphs and states------------------
# Define the Agent State, Edges and Graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

workflow = StateGraph(AgentState)
workflow.add_node("Bit_Support_Searcher", Bit_Support_Searcher_node)
workflow.add_node("Stock_Helper", Stock_Helper_node)
workflow.add_node("Insight_Researcher", Insight_Researcher_node)
workflow.add_node("supervisor", supervisor_chain)

# Define edges
for member in members:
    workflow.add_edge(member, "supervisor")

conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
workflow.set_entry_point("supervisor")

graph = workflow.compile()

# -----------------------------end of definition----------------------


# # Run the graph
# for s in graph.stream({
#     "messages": [HumanMessage(content="""איך אני יכול לקבל תמיכה?""")]
# , "chat_history":[]}):
#     if "__end__" not in s:
#         print(s)
#         print("----")



# final_response = graph.invoke({
#     "messages": [HumanMessage(
#         content="""Search for the latest AI technology trends in 2024,
#                 summarize the content
#                 and provide insights for each topic."""),"chat_history":[]]
# })

# print(final_response['messages'][1].content)



# {'question': "My name's bob. How are you?", 'chat_history': [HumanMessage(content="My name's bob. How are you?", additional_kwargs={}, example=False), AIMessage(content="I'm doing well, thank you. How can I assist you today, Bob?", additional_kwargs={}, example=False)], 'answer': "I'm doing well, thank you. How can I assist you today, Bob?"}

    
class Conversational:
    def __init__(self,graph=graph) -> None:
        self.graph=graph

    def run_graph(self,input_message:str,history=[]):
        # print("input_message",input_message)
        # history_trank=history
        response = graph.invoke({
            "messages": [HumanMessage(content=input_message)]
        })
        # return json.dumps(response['messages'][1].content, indent=2)
        print(response)
        return response['messages'][1].content

    def run_graphWithHistory(self,input_message:str,history=[]):
        # print("input_message",input_message)
        history_trank=history
        response = graph.invoke({
            "messages": [HumanMessage(content=input_message)],
            "chat_history":history_trank
        })
        # return json.dumps(response['messages'][1].content, indent=2)
        print(response)
        return response['messages'][1].content

    def __call__(self, context):
        # Extract the 'question' from the context dictionary
        question = context['question']
        chat_history = context['chat_history']
        if len(chat_history)==0:
            return f"{self.run_graph(question)}"
        else:
            return f"{self.run_graphWithHistory(question,chat_history)}"
        
       
    
if __name__ == "__main__":   
    
    print(Conversational()(
          {
              "question":"איך אני מקבל תמיכה?",
              "chat_history":[]
          })
    )
          

