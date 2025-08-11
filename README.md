# 🤖 GenAI Finance Chatbot

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python" alt="Python Version" />
  <img src="https://img.shields.io/badge/Streamlit-1.25%2B-orange?style=for-the-badge&logo=streamlit" alt="Streamlit Version" />
  <img src="https://img.shields.io/badge/LangChain-0.0.300%2B-green?style=for-the-badge" alt="LangChain Version" />
</p>

---

## Overview

**GenAI Finance Chatbot** is an intelligent, conversational AI chatbot designed to answer questions about financial data. It employs a Retrieval-Augmented Generation (RAG) architecture to provide accurate, context-aware responses based solely on a custom knowledge base contained in a CSV file.

The chatbot features a clean, interactive web interface built with Streamlit, supporting persistent chat sessions for a natural conversational experience.

---

## 🌟 Key Features

- 💬 **Interactive Chat Interface**  
  A clean and simple UI built with Streamlit for a natural conversational experience.

- 📚 **Custom Knowledge Base**  
  Answers questions based only on the information provided in your own CSV file.

- 🎯 **High Accuracy**  
  Uses RAG to minimize hallucinations by grounding responses in source documents.

- 🔄 **Persistent Conversation**  
  Remembers chat history during the session for continuity.

- 🔧 **Easy to Customize**  
  Replace the sample CSV file to tailor the chatbot to any domain.

---

## 🚀 Live Demo

*(Optional: Add a GIF or link to a live demo here)*

---

## ⚙️ How It Works (RAG Architecture)

The chatbot operates as an "open-book exam":

1. **Load & Index**  
   Loads `finance_knowledge_base.csv` using Pandas, converts question-answer pairs into embeddings with Sentence-Transformers, and stores them in a FAISS vector index.

2. **Retrieve**  
   Converts user queries into embeddings and retrieves semantically similar Q&A pairs using FAISS.

3. **Generate**  
   Feeds the query and retrieved context into a language model (`google/flan-t5-base`) via Transformers to generate human-readable answers grounded in the knowledge base.

4. **Orchestration**  
   The entire flow is managed using LangChain to seamlessly connect all components.

---

## ⚙️ Core Technologies

- **[Streamlit](https://streamlit.io/)** — Web app UI  
- **[Pandas](https://pandas.pydata.org/)** — Data loading and parsing  
- **[LangChain](https://github.com/hwchase17/langchain)** — RAG pipeline orchestration  
- **[FAISS](https://github.com/facebookresearch/faiss)** — Vector similarity search  
- **[Sentence-Transformers](https://www.sbert.net/)** — Text embeddings  
- **[Transformers](https://huggingface.co/transformers/)** — Language model inference  
- **[PyTorch](https://pytorch.org/)** — Deep learning backend  

---

## 📂 Project Structure
```plaintext

├── app.py # Main Streamlit application script
├── finance_knowledge_base.csv # Custom knowledge base CSV file
├── requirements.txt # Python dependencies
└── README.md # This documentation file
```


## 📦 Setup & Installation

1. **Clone the repository:**


   git clone <your-repo-url>
   cd <your-repo-name>

## Create and activate a virtual environment (recommended):


python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate


## Install dependencies:

pip install -r requirements.txt


# ▶️ How to Run
Make sure your knowledge base file finance_knowledge_base.csv is located in the project root directory.

Run the Streamlit app:


streamlit run app.py
Open your web browser at the URL provided by Streamlit (typically http://localhost:8501).


## Enjoy building with GenAI Finance Chatbot! 🚀
