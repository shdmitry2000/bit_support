import sys
import streamlit as st


import functools, operator, requests, os, json
from bs4 import BeautifulSoup
# from duckduckgo_search import DDGS
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain_community.llms import Anthropic

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain import hub

from langchain.agents.output_parsers import XMLAgentOutputParser

from langchain_anthropic import ChatAnthropic
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict, Union
import StockTool
from vectorDbbase import baseVebtorDb
# from vector_db_tools import *
from vector_dbs import *
import pandas as pd
from utility import syncdecorator
from langchain_core.runnables.config import (
    RunnableConfig,
)

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
)

import requests


from excelutility import getExcelDatainJson, load_data
import logging

from langchain_core.runnables import RunnableLambda ,RunnableParallel, RunnablePassthrough

NOT_FOUND_PROMPT="""
                     ניתן לפנות אלינו בטלפון
                     *6428
                    א'-ה' - 9:00-17:00 | ו' וערבי חג - 8:30-13:00      
                        """

class ClaudeTools():
    llmQA = ChatAnthropic(model='claude-3-sonnet-20240229', temperature=0)
    
    def __init__(self,fail_prompt) -> None:
        self.fail_prompt=fail_prompt
        corpus =  getExcelDatainJson()
        self.corpus_str = "\n".join([f"Question: {item['question']}\nAnswer: {item['answer']}" for item in corpus])

        # self.llmQA.add_instruction(corpus_str)
    
    def getLlm(self):
        return self.llmQA
        
    def question_answering_tool(self,input_str):
        result = self.llmQA([{"role": "user", "content": f"Question: {input_str}"}, {"role": "instruction", "content": self.corpus_str}])
   
        # result = self.llmQA(f"Question: {input_str}")
        if result.strip():
            return result.strip()
        else:
            return self.fail_prompt
        
        
           
    def getDataIncludeContextTool(self,name="Question Answering",description="Answer questions based on the loaded context"):
        # Create a custom tool instance
        return  Tool(name=name, description=description, func=self.question_answering_tool)

            # Logic for going from intermediate steps to a string to pass into model
    # This is pretty tied to the prompt
    def convert_intermediate_steps(self,intermediate_steps) -> List[BaseMessage]:
        log = ""
        print(intermediate_steps)
        for action, observation in intermediate_steps:
            log += (
                f"<tool>{action.tool}</tool><tool_input>{action.tool_input}"
                f"</tool_input><observation>{observation}</observation>"
            )
        return log
    
        messages = []
        for agent_action, observation in intermediate_steps:
            if isinstance(agent_action, OpenAIToolAgentAction):
                new_messages = list(agent_action.message_log) + [
                    _create_tool_message(agent_action, observation)
                ]
                messages.extend([new for new in new_messages if new not in messages])
            else:
                messages.append(AIMessage(content=agent_action.log))
            print("messages from tools:",messages)
        return messages


    
    def print_params(self,_dict):
        print(_dict)
        
       


    # Logic for converting tools to string to go in prompt
    def convert_tools(self,tools):
        return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])




    def getAgentwoPrompt(self,llm:ChatAnthropic,tools,verbose=True):
        prompt = hub.pull("hwchase17/xml-agent-convo")
        return self.getAgent(llm=llm,tools=tools,prompt=prompt,verbose=verbose)
        
    def getAgent(self,llm:ChatAnthropic,tools,prompt:ChatPromptTemplate,verbose=True):
        
        # return  createXmlAgent({ llm, tools, prompt });
        
        missing_vars = {"agent_scratchpad"}.difference(prompt.input_variables)
        if missing_vars:
            raise ValueError(f"Prompt missing required variables: {missing_vars}")
        
        
        # agent = (
        
        # {
        #     "messages": lambda x: x["messages"], # RunnableLambda(self.print_params)  , #|
        #     "agent_scratchpad": lambda x: self.convert_intermediate_steps(
        #         x["intermediate_steps"]
        #     ),
        # }
        # | prompt.partial(tools=self.convert_tools(tools))
        # | self.llmQA.bind(stop=["</tool_input>", "</final_answer>"])
        # | XMLAgentOutputParser()
        # )
        # return agent
    
    
    
        # llm_with_tools = self.llmQA.bind_tools(tools)
        
        agent = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: self.convert_intermediate_steps(
                    x["intermediate_steps"]
                )
            )
            | prompt.partial(tools=self.convert_tools(tools))
            | self.llmQA.bind(stop=["</tool_input>", "</final_answer>"])
            | XMLAgentOutputParser()
        )
        return agent



    
    def createPrompt(self,system_prompt)->ChatPromptTemplate:
        # systemPrompt="""
        # You are an expert in anthropic. 
        # Always answer questions starting with "As Dario Amodei told me". 
        # Respond to the following question:

        # Question: {question}
        # Answer:
        # "
        # 
        # 
        # ""
        # prompt = hub.pull("hwchase17/xml-agent-convo")
        # print(prompt)
        
        prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
                # MessagesPlaceholder(variable_name="chat_history"),  # Added chat history placeholder
            ])
        print (prompt)
        
        return prompt
        
        
        
    
    
    def getAgentExecuter(agent,tools):
        return  AgentExecutor(agent=agent, tools=tools)
        
        
        
# vectors.add_data()
# {'question': "My name's bob. How are you?", 'chat_history': [HumanMessage(content="My name's bob. How are you?", additional_kwargs={}, example=False), AIMessage(content="I'm doing well, thank you. How can I assist you today, Bob?", additional_kwargs={}, example=False)], 'answer': "I'm doing well, thank you. How can I assist you today, Bob?"}

    
class Conversational:
    


    def __init__(self,vector_db_factory=vectorDBLlama,myembeder=Embeddings.getdefaultEmbading()) -> None:
       
            # Initialize model
        llm = ChatOpenAI(model="gpt-4-turbo-preview",temperature=0.7)

        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
        
        self.myembeder=myembeder
        

        # # 1. Define custom tools
        #-------------bit tools ------------------
        bit_tool_description="Useful for founding answers for questions about bit application .Answer questions based on the loaded context"
        bit_tool_name="bit_FAQ_search"

        cloudeToolsDef=ClaudeTools(NOT_FOUND_PROMPT)
        tool_bit_searcher=cloudeToolsDef.getDataIncludeContextTool(name=bit_tool_name,description=bit_tool_description)
        toolsBit= [ tool_bit_searcher ]
        
        
       #--------------------------------------------------------------------
        
        # 2. Agents chatGPT
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
        
        def create_claude_agent(llm: ChatAnthropic, tools: list, system_prompt: str):
            prompt:ChatPromptTemplate=cloudeToolsDef.createPrompt(system_prompt)
            agent=cloudeToolsDef.getAgent(llm=cloudeToolsDef.getLlm(),tools=tools,prompt=prompt)
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


        Bit_Support_Searcher_agent = create_agent(llm, toolsBit,
                      """You are a support question answer searcher for bank application support. 
                      Based on the provided content first identify the list of topics, and what question represent every topic,
                      then use bit_FAQ_search tool to find an answer for every question for each topic one by one.
                      if you found answers be precision with answer and hold all data in answers include phone numbers , hours of work and amounts.    
                          """)
        Bit_Support_Searcher_node = functools.partial(agent_node, agent=Bit_Support_Searcher_agent, name="Bit_Support_Searcher")

        

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
        
