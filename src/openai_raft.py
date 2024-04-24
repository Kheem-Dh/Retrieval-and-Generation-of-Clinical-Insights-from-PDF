import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import GooglePalm
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Common function to create a vector store from a list of documents
def create_vector_store(documents, embedding_type="HuggingFace"):
    # Split documents into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # Select the appropriate embeddings
    if embedding_type == "OpenAI":
        api_key = os.getenv("OPENAI_API_KEY")
        embeddings = OpenAIEmbeddings(api_key=api_key)
    elif embedding_type == "HuggingFace":
        embeddings = HuggingFaceEmbeddings(model_name="multi-qa-MiniLM-L6-cos-v1")
    else:
        raise ValueError("Invalid embedding type. Choose 'OpenAI' or 'HuggingFace'.")

    # Create the Chroma vector store
    db = Chroma.from_documents(texts, embeddings)
    return db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# Function to initialize a QA system with a specified language model
def initialize_qa_system(retriever, model_type="OpenAI"):
    if model_type == "OpenAI":
        api_key = os.getenv("OPENAI_API_KEY")
        custom_model_name = os.getenv("FINE_TUNED_MODEL_NAME")
        llm = ChatOpenAI(temperature=0, model=custom_model_name, api_key=api_key)
    elif model_type == "GooglePalm":
        google_api_key = os.getenv("GOOGLE_API_KEY")
        llm = GooglePalm(api_key=google_api_key, temperature=0.1)
    else:
        raise ValueError("Invalid model type. Choose 'OpenAI' or 'GooglePalm'.")

    return RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )
