from vectorDbbase import Embeddings
# from vectorDBFaiss import faissRag ,getVectorDB
from vectorDBChroma import chromaRag,getVectorDB
# from vectorDBLlama import llamaRag,getVectorDB
# from vectorDBRedis import redisRag ,getVectorDB
class VectorDatabaseDAO:
    _instance = None
    vectordb=None

    def __new__(cls, embeder=Embeddings.getdefaultEmbading(), *args, **kwargs):
        if not cls._instance:
            cls._instance = super(VectorDatabaseDAO, cls).__new__(cls)
            cls._instance.__init__(embeder, *args, **kwargs)
        return cls._instance

    def __init__(self, embeder=Embeddings.getdefaultEmbading(), *args, **kwargs):
        self.embeder = embeder
        self.load_data(embeder)
        # Additional initialization logic here

    def load_data(self,embeder=Embeddings.getdefaultEmbading()):
        if self.vectordb is None:
            self.vectordb = getVectorDB(embeder)
        
    
    def getVector(self):
        return  self.vectordb

    def reset(self):
        self.vectordb = None
        self.load_data(self.embeder)


def check_embeding(ll,query):
    # ll.load()
    from bidi.algorithm import get_display


    print(ll.__class__.__name__,"answer for:\n",get_display(str(query)),"\n",
        get_display(str(ll.data_search(query))))
    
    
    # print('\n\n\n\n')
    # print(ll.__class__.__name__,"answer for:\n",get_display(str(query)),"\n",
    #     get_display(str(ll.query(query))))
    

def check_embeding_query(ll,query):
    # ll.load()
    from bidi.algorithm import get_display


   
    
    print('\n\n\n\n')
    print(ll.__class__.__name__,"answer for:\n",get_display(str(query)),"\n",
        ll.query(query))
    
def loadDB():
    # ll=faissRag(embedder=myembeder)
    # ll.delete()
    # ll.build_data(documents=document.get_documents())
    # ll.save() 

    # ll=RedisRag(embedder=myembeder)
    # ll.delete()
    # ll.build_data(documents=document.get_documents())
    # ll.save()

    # ll=llamaRag(embedder=myembeder)
    # ll.delete()
    # index=0
    # documents=[]
    # for item in document.faqs: 
    #     index=index+1
    #     document_tmp =llamaRag.createDocument(index=index,title=item['question'],text= " question : "+ str( item['question']) +" \n answer : "+ str(item['answer'])+ "\n")
    #                 # str(question_cleaned_string) +"\n"+ str(answer_cleaned_string))
    #     documents.append(document_tmp)     
    # ll.build_data(documents)
    # ll.save()
    pass
    
if __name__ == "__main__":         

    myembeder=Embeddings.getdefaultEmbading()

    vector= VectorDatabaseDAO(myembeder)

    # query = "האם אפשר לשלם חשבון בביט"
    # check_embeding(vector.getVector(),query)
 
    
    query = "איך אני מקבל תמיכה?"

    # check_embeding(vector,query)
    check_embeding_query(vector.getVector(),query)
    
    query = "מיהכן אוספים מטח?"

    # check_embeding(vector,query)
    check_embeding_query(vector.getVector(),query)
    
    query = "איך אני מוסיף כרטיס אשראי"
    # check_embeding(vector.getVector(),query)
    check_embeding_query(vector.getVector(),query)
    
    query = 'האם אפשר לשלם ארנונה ?'
    # check_embeding(vector.getVector(),query)
    check_embeding_query(vector.getVector(),query)
    
    query = 'איך אני מגיע לירח'
    # check_embeding(vector.getVector(),query)
    check_embeding_query(vector.getVector(),query)
    
    
    


    