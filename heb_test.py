
# import chromadb


import uuid
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from excelutility import load_data


# from chromadb.utils import embedding_functions

from vectorDbbase import *
from vectorDBChroma import chromaRag
from vectorDBFaiss import faissRag
from vectorDBLlama import llamaRag

# embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
#         # model_name="all-MiniLM-L6-v2")
#         model_name="dicta-il/dictabert-seg")

# class HebEncodingInstructions(embedding_functions.InstructorEmbeddingFunction):
#     # ef = embedding_functions.InstructorEmbeddingFunction(
#     model_name="dicta-il/dictabert-seg", device="cpu")


def getUniqueID():
        
    # Create a list of unique ids for each document based on the content
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in docs]
    unique_ids = list(set(ids))

    # Ensure that only docs that correspond to unique ids are kept and that only one of the duplicate ids is kept
    seen_ids = set()
    unique_docs = [doc for doc, id in zip(docs, ids) if id not in seen_ids and (seen_ids.add(id) or True)]

    # Add the unique documents to your database
    db = Chroma.from_documents(unique_docs, embeddings, ids=unique_ids, persist_directory='db')


# def getExcelDatainJson():
#     json_data =  load_data().to_dict(orient='records')
#     print("load additional questions:",json_data)
#     return json_data

def llamaIndexDocumentsFromJson(json_data):
    
    documents=[]
    for item in json_data: 
        document_tmp =llamaRag.createDocument(text= " question : "+ str( item['question']) +" \n answer : "+ str(item['answer'])+ "\n")
        documents.append(document_tmp)   
    return documents

def llamaIndexDocuments(text):
     
    documents=[]
    document_tmp =llamaRag.createTextNode(text= text)
    documents.append(document_tmp)   
    return documents
            
def createcromaIndexDocumentsFromJson(json_data):
    
    documents=[]
    for item in json_data: 
        document_tmp =chromaRag.createDocument(page_content=" question : "+ str( item['question']) +" \n answer : "+ str(item['answer'])+ "\n",
            metadata= "question"+ item['question']
        )
        documents.append(document_tmp)   
    return documents

def createcromaIndexDocuments(page_content,metadata=""):
    
    
    documents=[chromaRag.createDocument(page_content= page_content,metadata=metadata)]
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    
    
    return docs



def checkChromaVanila(query,page_content):
    import chromadb

    # client = chromadb.Client()
    from chromadb.db.base import UniqueConstraintError
    
    # client = chromadb.PersistentClient(path="chromadb/")  # data stored in 'db' folder
    client = chromadb.Client()  # data stored in 'db' folder

    embedding_function=Embeddings.getHebSentenceTransformerEmbeddingFunction()  
    
    try:
        collection = client.create_collection(name='test', embedding_function=embedding_function)
    except UniqueConstraintError:  # already exist collection
        collection = client.get_collection(name='test', embedding_function=embedding_function)
    
    
    
    
    collection.add(
        documents=[page_content],
        metadatas=[{"source": "my_source"}],
        ids=baseVebtorDb.getUUID_ID()
    )
    
    
    
    results = collection.query(
        query_texts=[query],
        n_results=2
    )
    

    # print results
    print("similar",results['documents'][0])



def checkChroma(query,page_content):
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_community.document_loaders import TextLoader
    # from langchain_community.embeddings.sentence_transformer import (
    #     SentenceTransformerEmbeddings,
    # )
    from langchain_community.vectorstores import Chroma


    # documents=[page_content]
    documents=[chromaRag.createDocument(page_content= page_content)]
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # embedding_function=HebEmbeddingFunction()
    
    embedding_function = Embeddings.getHebSentenceTransformerEmbeddings()
        # cache_dir="./SentenceTransformercache/",
        # model_name="dicta-il/dictabert-seg")  
        # model_name="all-MiniLM-L6-v2")


    # # load it into Chroma
    db = Chroma.from_documents(docs, embedding_function)

    # query it
    # query = "What did the president say about Ketanji Brown Jackson"
    # query ='How big is London'
    docs = db.similarity_search(query)

    # print results
    print("similar",docs[0].page_content)

    
    
def checkChromaReg(query,page_content):
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_community.document_loaders import TextLoader
    
    documents=[chromaRag.createDocument(page_content= page_content)]
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # embedding_function=HebEmbeddingFunction()
    

    
    embedding_function = Embeddings.getHebSentenceTransformerEmbeddings()
    
    cr=chromaRag(persist_dir="./test/chromadb/",embedder=embedding_function)
    
    cr.add_data(docs)
    print("similar data search",cr.data_search(query))
    
    
    print("simila otherr",cr.query(query))
    
def checkllamaReg(query,page_content):
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_community.document_loaders import TextLoader
    from langchain_community.embeddings.sentence_transformer import (
        SentenceTransformerEmbeddings,
    )
    
    embedding_function = Embeddings.getHebSentenceTransformerEmbeddings()
    
    cr=llamaRag(persist_dir="./test/chromadb/",embedder=embedding_function)
    document=cr.createDocument(index="1",title="",text= page_content)
    cr.add_data([document])
    print("similar data search",cr.data_search(query))
    
    
    # print("simila otherr",cr.query(query))
    
def checkFaissReg(query,page_content):
    
    
    embedding_function = Embeddings.getHebSentenceTransformerEmbeddings()
    
    cr=faissRag(persist_dir="./test/faisdb/",embedder=embedding_function)
    document=cr.createDocument(page_content= page_content)
    cr.add_data([document])
    print("similar data search",cr.data_search(query))
    
    
    print("simila otherr",cr.query(query))
    



def checkFaiss(query,page_content):
    from langchain_community.vectorstores import FAISS

    # load the document and split it into chunks
    # loader = TextLoader("./state_of_the_union.txt")
    # documents = loader.load()
    
   

    # documents=[page_content]
    documents=[chromaRag.createDocument(page_content= page_content)]
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    embedding_function = Embeddings.getHebSentenceTransformerEmbeddings()
    
    # # load it into Chroma
    db = FAISS.from_documents(docs, embedding_function)

    # query it
    # query = "What did the president say about Ketanji Brown Jackson"
    # query ='How big is London'
    docs = db.similarity_search(query)

    # print results
    print("similar",docs[0].page_content)


if __name__ == "__main__":  
    embedng=Embeddings.getHebChromaEmbeddingFunction()
    
    query='How big is London'
    page_content="London has 9,787,426 inhabitants at the 2011 census . London is known for its finacial district"
    
    # query_embedding = embedng(query)
    passage_embedding = embedng(['London has 9,787,426 inhabitants at the 2011 census',
                                    'London is known for its finacial district'])
    
    query='האם אפשר לשלם חשבון בביט?'
    page_content="עם bit תוכלו לקבל בקשות תשלום מהעירייה ישירות לנייד, תוכלו לאשר את התשלומים בלחיצת כפתור ולעקוב אחריהם בשיא הנוחות. כדי להצטרף לשירות, סורקים באפליקציית bit את קוד ה-QR שמופיע בחשבון לתשלום, משלמים ומאשרים הצטרפות לשירות"
    # query_embedding = embedng(query)
    # passage_embedding = embedng(page_content)
    # print("Similarity:", util.dot_score(query_embedding, passage_embedding))

    # checkChromaVanila(query,page_content)
    
    checkChroma(query,page_content)
    
    # checkFaiss(query,page_content)
    
    # checkChromaReg(query,page_content)
    
    # checkFaissReg(query,page_content)
    
    
    # checkllamaReg(query,page_content)
    
    exit(0)
    
    # create the open-source embedding function
        

    # print("query_embedding",query_embedding)

