<h1 align="center">Retrieval and Generation of Clinical Insights from PDF</h1>

<p align="center">
    <img src="https://readme-typing-svg.demolab.com?font=Georgia&size=22&duration=5000&pause=5000&color=007BFF&background=00000000&center=true&vCenter=true&width=435&lines=Retrieval-Augmented+Fine-Tuning+System;Unified+Question+Answering+Framework" alt="Typing SVG" />
</p>

<p align="center">
    <img alt="Python Version" src="https://img.shields.io/badge/python-3.11.5-green.svg">
    <img alt="Last Commit" src="https://img.shields.io/badge/last_commit-April_2024-green.svg">
    <img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg">
    <img alt="LangChain" src="https://img.shields.io/badge/LangChain-0.1.16-blue.svg">
    <img alt="ChromaDB" src="https://img.shields.io/badge/ChromaDB-0.4.24-orange.svg">
    <img alt="Google Palm" src="https://img.shields.io/badge/Google-Palm-FFCC00.svg">
</p>


## Table of Contents
- [Overview](#overview)
- [Features](#features)
  - [Unified Vector Store](#1-unified-vector-store)
  - [Dynamic LLM Integration](#2-dynamic-llm-integration)
  - [Interactive Streamlit UI](#3-interactive-streamlit-ui)
- [System Requirements](#system-requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)
- [Contributions](#contributing)

## Overview
The RAFT project integrates Retrieval-Augmented Generation (RAG) with fine-tuning approaches to provide a versatile question-answering system. Leveraging the strengths of OpenAI GPT-3.5 Turbo and Google MedPaLM 2, this framework aims to enhance information retrieval and generation capabilities across various domains.

## Features
### 1. Unified Vector Store
A single vector store is used for document retrieval, ensuring efficiency and speed regardless of the selected generation model.

### 2. Dynamic LLM Integration
Seamlessly switch between OpenAI GPT-3.5 Turbo and MedPaLM 2 depending on the specific needs of the query and the context provided by the user.

### 3. Interactive Streamlit UI
A user-friendly interface allows for real-time uploading of PDF documents and querying, making the system accessible to users with minimal technical background.

## System Requirements
- Python 3.11.5 or higher
- Dependencies as listed in the `requirements.txt` file

## Setup
To get started with the RAFT project, follow these steps:
1. **Clone the repository**:

```bash
git clone <repository_url>
```
2. **Navigate to the project directory**:
```bash
cd RAFT
```

3. **Create a Virtual Environment:**
```bash
python -m venv venv
```
4. **Activate the Virtual Environment:**
```bash
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

5. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

6. **Set Environment Variables:**
Create a .env file in the project root with the following variables (Please refer the `example.env` file):
```bash
OPENAI_API_KEY=your_openai_api_key_here
FINE_TUNED_MODEL_NAME=ft:gpt-3.5-turbo-0125:your_fine_tuned_model_name
GOOGLE_API_KEY=your_google_api_key_here
```

## Usage
To run the Streamlit application, use the following command:
```bash
streamlit run app.py
```

This will launch the Streamlit server, allowing you to upload PDF files and ask questions in real-time.

## Uploading PDF Files
* Use the file uploader in the Streamlit app to upload one or more PDF files.
* The system will process the PDFs and create a vector store for retrieval.
## Asking Questions
* After uploading PDF files, enter your question in the text area and click the "Ask" button.
* The system will return an answer based on the selected language model.
## Acknowledgments
* LangChain: A framework for building applications with large language models and data.
* Chroma: An open-source embedding database for retrieval.
* OpenAI: Provides the GPT-3.5 Turbo language model.
* Google Palm: A generative language model from Google.
## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request for any changes or improvements.