################################load pdf ###########################################
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

pc=Pinecone(api_key="your api key")
os.environ["PINECONE_API_KEY"] = "your api key"
OPENAI_KEY="your api key"

index_name="first-indexnew3"
def Add_vector_data_to_db(file_name:str):
    global index_name
    # load_docs=PyPDFLoader("cp.pdf")
    load_docs=PyPDFLoader(file_name)
    docs=load_docs.load()
    ########################text spliter from lanchain # Initialize Pinecone######################
    text_spliter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=30)
    splts=text_spliter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY,model="text-embedding-3-small")
    index=pc.Index(name=index_name,pool_threads=50,connection_pool_maxsize=50)
    if not index:
        # pc.delete_index(index_name)
        # print("deleted old dimension index ")
        # pc.create_index(name="first-indexnew2",dimension=3072,metric="cosine")
        #create new index with index_name
        pc.create_index(name=index_name,dimension=1536,metric="cosine",spec=ServerlessSpec(cloud="aws",region="us-east-1",),deletion_protection="disabled")
    vectorstore = PineconeVectorStore.from_documents(documents=splts,embedding=embeddings,index_name=index_name)
    print("vector added to db for file",file_name)
# Build retriever
def ask_question(question:str):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY, model="text-embedding-3-small")
    # retriever = vectorstore.as_retriever()
    vectorstore=PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    class RetrieverRunnable:
        def __init__(self, retriever):
            self.retriever = retriever
        def __call__(self, input_data):
            # Extract question string
            if isinstance(input_data, dict):
                query = input_data.get("question", "")
            else:
                query = input_data
            docs = self.retriever.get_relevant_documents(query)
            return "\n".join(doc.page_content for doc in  docs)
    retriever_runnable = RetrieverRunnable(retriever)
    ##########################################pull langchian prompt and use it Load prompt & LLM####
    from langchain import hub
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(openai_api_key=OPENAI_KEY)
    # Build RAG chain
    rag_chain = ({"context": retriever_runnable, "question": RunnablePassthrough()}| prompt| llm| StrOutputParser())
    # Query example
    # query = "can you tell types of computer"
    # query=input("what is your question:")
    query=question
    answer = rag_chain.invoke({"question": query})
    return {"Question:":query,"Answer:":answer}

print(ask_question("can you tell me about what is cpu ?"))

#this fun to update pdf in vector pincone db 

# Add_vector_data_to_db("sample.pdf")