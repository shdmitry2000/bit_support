import sys ,logging
from langchain.agents import initialize_agent,Tool
# from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.tools import tool
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent
from langchain_community.llms import OpenAI
import yfinance as yf
import baseToolsHandler

load_dotenv()

# Get the prompt to use
prompt = hub.pull("hwchase17/react")


class StockTool(baseToolsHandler.projectBaseTool):

    def __init__(self,memory_name="stock_data", withmemory=False) -> None:
        super().__init__(withmemory)
        # OpenAI.api_key =os.environ["OPENAI_API_KEY"]
        
# Define a tool

    # @tool("my_tool", return_direct=False)
    # def my_tool(input_dict: dict) -> str:
    #     """Processes multiple inputs within a single input."""
    #     # Extract the inputs from the input_dict
    #     input1 = input_dict['input1']
    #     input2 = input_dict['input2']
    #     print("input1",input1,"input1",input2)
        

    #     # Process the inputs...
    #     # ...

    #     # Return the result
    #     return "Result"



    def get_stock_price(self,symbol):
        print("get ticker price for",symbol)
        ticker = yf.Ticker(symbol)
        todays_data = ticker.history(period='1d')
        return round(todays_data['Close'][0], 2)



    def getToolDefinition(self):
        
        StockPriceTool = Tool(
            name='Get Stock Ticker price',
            func= self.get_stock_price,
            # func=lambda q: str(self.get_stock_price(q)),
            description="Useful for when you need to find out the price of stock. You should input the stock ticker used on the yfinance API"
        )

        return StockPriceTool
    
        

    def getExecuter(self):
        # Initialize the agent
        llm = ChatOpenAI(model="gpt-4-turbo-preview")
        tools=[self.getToolDefinition()]
        
        agent = initialize_agent(llm=llm,tools= tools,verbose=True,handle_parsing_errors=True)
        # agent = create_react_agent(llm, [my_tool,getToolDefinition()], prompt)

        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
        
        return  agent


if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


    print(StockTool().getExecuter().run("What is the price of Google stock"))

    # print(agent.invoke({'input':"What is the price of Google stock",'chat_history':[]}))
