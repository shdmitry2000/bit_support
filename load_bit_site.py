# # Load HTML
import json
from llama_index.schema import Document

import os
from bs4 import BeautifulSoup
from langchain_community.document_loaders import AsyncChromiumLoader
import re
from langchain_openai import OpenAI
# from langchain.index import LlamaIndex
from llama_index import LLMPredictor, download_loader, VectorStoreIndex, ServiceContext
from llama_index import GPTVectorStoreIndex, ServiceContext ,Document 
from llama_index import download_loader


from llama_index import PromptHelper
from dotenv import load_dotenv
import vector_dbs

load_dotenv()


loader = AsyncChromiumLoader(["https://www.bitpay.co.il/he/private-faq"])
html_content = loader.load()
soup = BeautifulSoup(html_content[0].page_content, "html.parser")
# Find all .faq-single.get-func divs
faq_divs = soup.select('.faq-single.get-func')

# List to store all FAQ pairs
faqs = []
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
    
    
    index=index+1
    document = Document(id=index, title=question, 
                        text= " question : "+ str( question) +" \n answer : "+ str(answer)+ "\n" )
                        # str(question_cleaned_string) +"\n"+ str(answer_cleaned_string))
    documents.append(document)
    
    
    # text= json.dumps(faq_pair),
    # embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002", openai_api_version="2023-05-15")
    # docsearch = Chroma.add_texts(text, embeddings)
    
    # faqs.append(faq_pair)
    faqs.append(faq_pair_cleaned_string)

    
# create_llama_index(documents)
# create_faiss_index(documents)
# create_croma_index(documents)



ll=vector_dbs.llamaRag(embedder=vector_dbs.Embeddings.getdefaultEmbading())
ll.delete()
ll.build_data(documents)

ll.save()

# Query the index
query = "מיהכן אוספים מטח?"

    
query_engine = ll.as_query_engine()
response = query_engine.query(query)
from bidi.algorithm import get_display

print(get_display(str(response)))


query = "מי נותן תמיכה?"

    
response = query_engine.query(query)
print(get_display(str(response)))


# def create_croma_index(ask):
#     fdb= indexdb.Chromaindexdb()
    



#     from langchain_core.documents import Document
#     Documents=[]
#     for d in ask:
#         Documents.append(Document(page_content=d.text,metadata={'source': '/Users/dmitryshlymovich/workspace/chatgpt/tests/XpayFinance_LLM/data/data.json'}))
        

   
    
#     fdb.create_db(Documents)
    
    
#     from bidi.algorithm import get_display

    
    
    
#     query = "מיהכן אוספים מטח?"

#     faissdb=fdb.getDb()
    
#     docs = faissdb.similarity_search(query)
#     print("answer",get_display(str(docs[0])))

      
#     # from bidi.algorithm import get_display

#     # print(get_display(str(response)))


#     query = "מי נותן תמיכה?"

        
#     docs = faissdb.similarity_search(query)
#     print("answer",get_display(str(docs[0])))
    
# def create_faiss_index(ask):
#     fdb= indexdb.Faissindexdb()
    
#     # data_dict = json.loads(faq_pair)

#     # JsonDataReader = download_loader("JsonDataReader")
#     # loader = JsonDataReader()
#     # documents = loader.load_data(faq_pair)
#     # index = VectorStoreIndex.from_documents(documents)


#     # from langchain_core.documents import Document
#     # Documents=[]
#     # for d in ask:
#     #     Documents.append(Document(page_content=d.text,metadata={'source': '/Users/dmitryshlymovich/workspace/chatgpt/tests/XpayFinance_LLM/data/data.json'}))
        

   
    
#     # fdb.create_db(Documents)
    
    
#     from bidi.algorithm import get_display

    
    
    
#     query = "מיהכן אוספים מטח?"

#     faissdb=fdb.getDb()
    
#     docs = faissdb.similarity_search(query)
#     print("answer",get_display(str(docs[0])))

      
#     # from bidi.algorithm import get_display

#     # print(get_display(str(response)))


#     query = "מי נותן תמיכה?"

        
#     docs = faissdb.similarity_search(query)
#     print("answer",get_display(str(docs[0])))
    
    
