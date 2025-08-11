# Finance-Chatbot
# GenAI Finance Chatbot 🤖
This project is an intelligent, conversational AI chatbot designed to answer questions about financial data. It uses a Retrieval-Augmented Generation (RAG) architecture to provide accurate, context-aware answers based on a custom knowledge base provided in a CSV file.

The application is built with a user-friendly web interface using Streamlit, allowing for interactive and persistent chat sessions.

Key Features
Interactive Chat Interface: A clean and simple UI built with Streamlit for a natural conversational experience.

Knowledge Base Integration: Answers questions based only on the information provided in your own CSV file.

High Accuracy: Leverages the RAG model to minimize "hallucinations" (incorrect or made-up answers) by grounding responses in the source documents.

Persistent Conversation: Remembers the chat history for the duration of the session.

Easy to Customize: Simply replace the sample CSV file with your own data to create a chatbot for any domain.

How It Works (RAG Architecture)
The chatbot does not rely on pre-trained knowledge for its specific answers. Instead, it follows an "open-book exam" process:

Load & Index: The application starts by reading the finance_knowledge_base.csv file using Pandas. Each question-answer pair is then converted into a numerical representation (embedding) using Sentence-Transformers. These embeddings are stored in a highly efficient FAISS vector index.

Retrieve: When you ask a question, the system converts your query into an embedding and uses FAISS to find the most semantically similar Q&A pairs from the knowledge base.

Generate: The original question and the retrieved context are passed to a Large Language Model (google/flan-t5-base via the Transformers library). The model then generates a human-readable answer based only on the provided information.

Orchestration: The entire pipeline—from data loading to response generation—is managed and connected using the LangChain framework.

Core Technologies
This project is built with the following key libraries:

streamlit: For creating the interactive web application and user interface.

pandas: For loading and parsing the CSV knowledge base.

langchain: The core framework for orchestrating the RAG pipeline.

faiss-cpu: For creating the efficient in-memory vector store.

sentence-transformers: For generating the text embeddings.

transformers: For loading and running the open-source language model from Hugging Face.

torch: The underlying deep learning framework required by transformers.

Setup & Installation
Clone the repository:

git clone <your-repo-url>
cd <your-repo-name>

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required packages:

pip install -r requirements.txt

How to Run
Make sure your knowledge base file, finance_knowledge_base.csv, is in the root directory of the project.

Run the Streamlit application from your terminal:

streamlit run app.py

Open your web browser to the local URL provided by Streamlit.
