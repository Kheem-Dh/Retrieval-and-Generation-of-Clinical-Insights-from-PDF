import streamlit as st
import os
from dotenv import load_dotenv
from src.openai_raft import create_vector_store, initialize_qa_system
from src.medpalm_raft import process_pdfs, initialize_medpalm_qa
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(
    page_title="Unified Retrieval and Generation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Retrieval and Generation of Clinical Insights from PDFs")
# Ensure Google API key is set
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    st.error("Google API key is missing from environment variables.")
    st.stop()  # Stop Streamlit if the key is missing

# Sidebar setup with image and language model selection
st.sidebar.image("./data/processed/logo.jpg", caption="Clinical Pearl", use_column_width=True) 
st.sidebar.title("Language Model Selection")
llm_choice = st.sidebar.radio(
    "Choose a language model for generation:",
    ("OpenAI GPT-3.5 Turbo", "MedPaLM 2")
)

# File uploader for PDFs
uploaded_files = st.file_uploader("Upload your PDF files here:", accept_multiple_files=True, type=["pdf"])

# Ensure the upload directory exists
upload_dir = "data/uploaded_pdfs"
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

# Check if files are uploaded
if uploaded_files:
    st.info("Processing uploaded files...")

    # List to store file paths for uploaded documents
    uploaded_file_paths = []

    # Save the uploaded files to disk
    for uploaded_file in uploaded_files:
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())  # Write the uploaded content to a local file
        uploaded_file_paths.append(file_path)  # Store the local file path

    # Process PDFs and create the vector store
    retriever = process_pdfs(uploaded_file_paths)  # Pass the local file paths to `process_pdfs`

    # Initialize the QA system with the selected LLM
    if llm_choice == "OpenAI GPT-3.5 Turbo":
        qa_system = initialize_qa_system(retriever, model_type="OpenAI")
    elif llm_choice == "MedPaLM 2":
        try:
            qa_system = initialize_medpalm_qa(retriever, google_api_key=google_api_key)
        except Exception as e:
            st.error(f"Error initializing MedPaLM QA system: {e}")
            st.stop()


    st.success("PDF files processed successfully. You can now ask questions.")

    # User input for questions
    user_query = st.text_area("Enter your question:", height=100)

    # Button to submit the query
    if st.button("Ask"):
        try:
            result = qa_system({"query": user_query})
            response = result["result"]
            st.success(f"Response: {response}")
        except Exception as e:
            st.error(f"An error occurred during QA retrieval: {e}")
else:
    st.info("Please upload one or more PDF files to begin.")