

import os
from dotenv import load_dotenv
import pandas as pd


load_dotenv()
EXCEL_FILE = os.getenv("EXCEL_FILE")

def load_data(file_path = EXCEL_FILE) :

    # Check if the file exists, if not, create it with the specified columns
    if not os.path.isfile(file_path):
        df = pd.DataFrame(columns=['question', 'answer'])
        df.to_excel(file_path, index=False)
    else:
        df =  pd.read_excel(file_path)
        return df


def getExcelDatainJson():
    json_data =  load_data().to_dict(orient='records')
    # print("load additional questions:",json_data)
    return json_data
# df= load_data()

def save_data(json_data,file_path=EXCEL_FILE):
    df = pd.json_normalize(json_data) # Uncomment if your data is not already in DataFrame format

    # Save DataFrame to Excel
    df.to_excel(file_path, index=False)
    