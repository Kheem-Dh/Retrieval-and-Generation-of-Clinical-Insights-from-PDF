import os
import pickle
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import GooglePalm  
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Function to process PDF files and create a vector store
def process_pdfs(uploaded_files, embedding_model_name="multi-qa-MiniLM-L6-cos-v1"):
    documents = []
    # Load PDFs and extract content
    for uploaded_file in uploaded_files:
        pdf_loader = PyPDFLoader(uploaded_file)
        documents.extend(pdf_loader.load())
    
    # Split documents into smaller text chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # Create embeddings and build a Chroma vector store
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    db = Chroma.from_documents(texts, embeddings)

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    
    return retriever

def initialize_medpalm_qa(retriever, google_api_key):
    # Review this initialization
    llm = GooglePalm(api_key=google_api_key, temperature=0.1)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa
