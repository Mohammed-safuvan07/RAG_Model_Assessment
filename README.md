


<img width="1361" height="686" alt="Screenshot_RAG" src="https://github.com/user-attachments/assets/cd981750-9919-43c1-89e4-1fb32b52c7e4" />




RAG PDF Chatbot
Overview

This project is a Retrieval-Augmented Generation (RAG) PDF Chatbot built using Streamlit, LangChain, FAISS, Hugging Face Embeddings, and Google Gemini.

The application allows users to upload one or more PDF documents and ask questions based on the uploaded content. The chatbot retrieves the most relevant information from the documents and generates accurate responses using Google's Gemini model.

Features
* Upload multiple PDF documents
* Extract text from PDF files
* Split text into manageable chunks
* Generate embeddings using Hugging Face
* Store embeddings in a FAISS vector database
* Retrieve relevant document chunks using similarity search
* Generate answers using Google Gemini
* Simple and interactive Streamlit interface

Technologies Used

* Python
* Streamlit
* LangChain
* FAISS
* Hugging Face Embeddings (sentence-transformers/all-MiniLM-L6-v2)
* Google Gemini
* PyPDF2
* Project Workflow
* Upload one or more PDF files.
* Extract text from the uploaded PDFs.
* Split the extracted text into overlapping chunks.
* Generate vector embeddings for each chunk.
* Store the embeddings in a FAISS vector database.
* Enter a question related to the uploaded documents.
* Retrieve the most relevant chunks using similarity search.
* Send the retrieved context to Gemini.
* Display the generated answer.

Project Structure
Chatbot_RAG/
│── app.py
│── requirements.txt
│── runtime.txt
│── Procfile
│── README.md
│── faiss_index/
Setup Instructions

1. Install Dependencies
pip install -r requirements.txt
2. Run the Application
streamlit run app.py
3. Using the Application
Enter your Google Gemini API Key.
Upload one or more PDF files.
Click Submit & Process.
Wait until the documents are processed.
Ask questions related to the uploaded PDFs.
