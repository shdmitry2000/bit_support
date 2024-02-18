from vectorDbbase import Embeddings
from vectorDBFaiss import faissRag
from vectorDBChroma import chromaRag
from vectorDBLlama import llamaRag
# from vectorDBRedis import redisRag






def check_embeding(ll,query):
    # ll.load()
    from bidi.algorithm import get_display


    print(ll.__class__.__name__,"answer for:\n",get_display(str(query)),"\n",
        get_display(str(ll.data_search(query))))
    
# def check_embeding2(ll,query):
#     # ll.load()
#     from bidi.algorithm import get_display


#     print(ll.__class__.__name__,"answer for:\n",get_display(str(query)),"\n",
#             get_display(str(ll.getRetriverChain().invoke(query))))
    
# def check_embeding3(ll,query):
#     # ll.load()
#     from bidi.algorithm import get_display

#     (agent,executer)=ll.getRetriverAgent()
#     print(ll.__class__.__name__,"answer for:\n",get_display(str(query)),"\n",
#             get_display(str(executer.invoke({"input": query}))))
    


# def check_embeding1(ll,query,history=[]):
#     # ll.load()
#     from bidi.algorithm import get_display


#     print(ll.__class__.__name__,"answer for:\n",get_display(str(query)),"\n",
#             get_display(str(ll.getRetriverChain().invoke({
#         "question": query,
#         "chat_history": history,
#     }))))
    
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

    


    query = "איך אני מקבל תמיכה?"

    # check_embeding(chromaRag(embedder=myembeder),query)
    # check_embeding(RedisRag(embedder=myembeder),query)
    check_embeding(llamaRag(embedder=myembeder),query)

    query = "מיהכן אוספים מטח?"

    # check_embeding(chromaRag(embedder=myembeder),query)
    # check_embeding(RedisRag(embedder=myembeder),query)
    check_embeding(llamaRag(embedder=myembeder),query)


    