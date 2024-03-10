import gc
from utility import profile
from vectorDbbase import Embeddings
import vectorDBFaiss
import vectorDBChroma
import vectorDBChroma
import vectorDBRedis
import vectorDBLlama 

# from vectorDBFaiss import faissRag ,getVectorDB
# from vectorDBChroma import chromaRag,getVectorDB
# from vectorDBLlama import llamaRag,getVectorDB
# from vectorDBRedis import redisRag ,getVectorDB



class VectorDatabaseDAO:
    _instance = None
    vectordb = None

    def __new__(cls, vector_db_factory,embeder=Embeddings().getdefaultEmbading(), *args, **kwargs):
        if not cls._instance:
            cls._instance = super(VectorDatabaseDAO, cls).__new__(cls)
            cls._instance.__init__(vector_db_factory,embeder, *args, **kwargs)
        return cls._instance

    def __init__(self, vector_db_factory, embeder=Embeddings().getdefaultEmbading(), *args, **kwargs):
        self.embeder=embeder
        self.vector_db_factory = vector_db_factory
        self.load_data(self.embeder)

    @profile
    def load_data(self,embeder=Embeddings().getdefaultEmbading()):
        if self.vectordb is None:
            print("load vectors began")
            self.vectordb = self.vector_db_factory.getVectorDB(embeder)
            gc.collect()
            print("gc empted")
        
    
        

    def getVector(self):
        return self.vectordb

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
    

if __name__ == "__main__":         

    myembeder=Embeddings().getdefaultEmbading()

    vector= VectorDatabaseDAO(vector_db_factory= vectorDBLlama,embeder= myembeder)

    # query = "האם אפשר לשלם חשבון בביט"
    # check_embeding(vector.getVector(),query)
 
    
    # query = "איך אני מקבל תמיכה?"

    # # check_embeding(vector,query)
    # check_embeding_query(vector.getVector(),query)
    
    # query = "מיהכן אוספים מטח?"

    # # check_embeding(vector,query)
    # check_embeding_query(vector.getVector(),query)
    
    # query = "איך אני מוסיף כרטיס אשראי"
    # # check_embeding(vector.getVector(),query)
    # check_embeding_query(vector.getVector(),query)
    
    # query = 'האם אפשר לשלם ארנונה ?'
    # # check_embeding(vector.getVector(),query)
    # check_embeding_query(vector.getVector(),query)
    
    query = 'איך אני מגיע לירח'
    # check_embeding(vector.getVector(),query)
    check_embeding_query(vector.getVector(),query)
    
    
    


    