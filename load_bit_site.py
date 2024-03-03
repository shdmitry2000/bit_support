# # Load HTML
import json


import os
from bs4 import BeautifulSoup
from langchain_community.document_loaders import AsyncChromiumLoader
import re
# from langchain_openai import OpenAI
# from langchain.index import LlamaIndex
# from llama_index import LLMPredictor, download_loader, VectorStoreIndex, ServiceContext
# from llama_index import GPTVectorStoreIndex, ServiceContext ,Document 
# from llama_index import download_loader


# from llama_index import PromptHelper
from dotenv import load_dotenv
# import vector_faiss_dbs
from excelutility import *

load_dotenv()


loader = AsyncChromiumLoader(["https://www.bitpay.co.il/he/private-faq"])
html_content = loader.load()
soup = BeautifulSoup(html_content[0].page_content, "html.parser")
# Find all .faq-single.get-func divs
faq_divs = soup.select('.faq-single.get-func')

# List to store all FAQ pairs
faqs = []
faqsc = []
documents=[]
index=0

for div in faq_divs:
    question = div.select_one('.question').get_text(strip=True)
    question_cleaned_string = question.replace('"', '')
    question_cleaned_string = question_cleaned_string.replace('\xa0', "")
    question_cleaned_string = question_cleaned_string.replace('/', "").encode('utf-8')
    answer = div.select_one('.answer .cmsn-table').get_text(strip=True)
    answer_cleaned_string =answer.replace('"', '')
    answer_cleaned_string =answer_cleaned_string.replace("'", '')
    answer_cleaned_string = answer_cleaned_string.replace('\xa0', "")
    answer_cleaned_string = answer_cleaned_string.replace('/', "").encode('utf-8')


    faq_pair = {
        "question": question,
        "answer": answer
    }
    
    
    faq_pair_cleaned_string = {
        "question": question_cleaned_string,
        "answer": answer_cleaned_string
    }
    
    
    # index=index+1
    # document = Document(id=index, title=question, 
    #                     text= " question : "+ str( question) +" \n answer : "+ str(answer)+ "\n" )
    #                     # str(question_cleaned_string) +"\n"+ str(answer_cleaned_string))
    # documents.append(document)
    
    
    # text= json.dumps(faq_pair),
    # embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002", openai_api_version="2023-05-15")
    # docsearch = Chroma.add_texts(text, embeddings)
    
    # faqs.append(faq_pair)
    faqsc.append(faq_pair_cleaned_string)
    
    faqs.append(faq_pair)

from vectorDBLlama import llamaRag,loadLlamaDB,Embeddings
from vectorDBFaiss import faissRag,loadfaissDB
from vectorDBChroma import chromaRag,loadChromaDB



myembeder=Embeddings.getdefaultEmbading()


# loadfaissDB(faqs,myembeder)
# loadLlamaDB(faqs,myembeder)
    
# loadChromaDB(faqs,myembeder)
# loadRedisDB(faqs,myembeder)

save_data(faqs)


# import csv
# from typing import List, Tuple

# def get_csv() -> List[Tuple[str, str]]:
#     csv_data = [
#         (doc["question"], doc["answer"]) for doc in faqs
#     ]
#     return csv_data

# def save_to_csv(data: List[Tuple[str, str]], filename: str):
#     with open(filename, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['Question', 'Answer'])  # Write the header
#         writer.writerows(data)  # Write the data rows

# # Example usage:
# csv_data = get_csv()
# save_to_csv(csv_data, 'output.csv')

