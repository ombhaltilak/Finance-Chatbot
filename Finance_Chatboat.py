# app.py

import streamlit as st
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.docstore.document import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import os

# --- UI CONFIGURATION ---
st.set_page_config(
    page_title="GenAI Finance Chatbot",
    page_icon="🤖",
    layout="wide"
)

# --- ONE-TIME SETUP AND CACHING ---
@st.cache_resource
def setup_rag_chain(csv_path):
    """
    Sets up the entire RAG pipeline using a CSV file as the knowledge base.
    """
    # 1. Load Knowledge Base from CSV
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"Error: The file '{csv_path}' was not found. Please make sure it's in the same directory as app.py.")
        return None

    # 2. Create LangChain Documents from the CSV data
    docs = []
    for index, row in df.iterrows():
        if 'Question' in row and 'Answer' in row and pd.notna(row['Question']) and pd.notna(row['Answer']):
            content = f"Question: {row['Question']}\nAnswer: {row['Answer']}"
            docs.append(Document(page_content=content))

    if not docs:
        st.error("Could not create any documents from the CSV. Please check that it contains 'Question' and 'Answer' columns with data.")
        return None

    # 3. Create Embeddings and Vector Store (FAISS)
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.from_documents(docs, embeddings)

    # 4. Set up the Large Language Model (LLM)
    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=1000
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # 5. Create the RAG Chain
    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )
    return qa_chain

# --- STREAMLIT UI ---

st.title("🤖 GenAI Finance Chatbot")

# Sidebar for information and actions
with st.sidebar:
    st.header("About")
    st.info(
        "This is an intelligent assistant for finance queries using a RAG architecture. "
        "It leverages LangChain, Hugging Face Transformers, and FAISS for efficient and accurate responses."
    )
    # Button to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Define the path to your knowledge base file
KNOWLEDGE_BASE_FILE = "finance_knowledge_base.csv"

# Load the RAG chain (this will be fast due to caching)
qa_chain = setup_rag_chain(KNOWLEDGE_BASE_FILE)

# Handle new chat input
if prompt := st.chat_input("Ask a question about your finances..."):
    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and display assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if qa_chain:
                try:
                    result = qa_chain({"query": prompt})
                    response = result["result"]
                    st.write(response)
                    # Add assistant response to session state
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_message = f"An error occurred: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
            else:
                st.error("Chatbot is not initialized. Please check the knowledge base file.")

