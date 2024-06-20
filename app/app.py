from dotenv import load_dotenv
import os 
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai.chat_models.base import ChatOpenAI
import time


DATA_FOLDER=os.getenv("DATA_FOLDER", default = "app/data")

def get_rag_chain():
    print("get_rag_chain_from_csv called")
    text_chunks = get_chunks()
    vector_db = get_vector_db(text_chunks, False)
    chain = get_conversation_chain(vector_db)
    return chain

def get_chunks():
    print("get chunks called")
    loader = DirectoryLoader(os.path.join(os.getcwd(), DATA_FOLDER), loader_cls=PyPDFLoader)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", "\t"])
    text_chunks = text_splitter.split_documents(pages)
    return text_chunks

def get_vector_db(text_chunks, reload):
    print("Create embeddings")
    start_time = time.time()
    #embeddings = llm_client.get_embeddings()
    embeddings = OpenAIEmbeddings() 
    print(f"Time taken to create embeddings: {time.time() - start_time} sec")
    start_time = time.time()
    path = "./indexes/data_index"
    if os.path.exists(path) and not reload:
        print(f"Loading vector index from path: {path}")
        faiss_db = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        print(f"Time taken to load vector index: {time.time() - start_time} sec")
    else:
        faiss_db = FAISS.from_documents(text_chunks, embeddings)
        faiss_db.save_local(path)
        print(f"Time taken to create vector index: {time.time() - start_time} sec")
    return faiss_db

def get_conversation_chain(vector_db):
    print("get conversation chain called")
    #llm = llm_client.create_llm_client()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                 temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        memory=memory,
        #callbacks=[handler]
        # return_source_documents = True
    )
    return conversation_chain


def handle_user_question(user_question, mychain):
    print("handle user question called")
    result = mychain({"question": user_question})
    print(result)

    return

def main():
    print("Main method called")
    load_dotenv()
    chain = get_rag_chain()
    handle_user_question("What is backup frequency?", chain)


if __name__ == '__main__':
    main()
