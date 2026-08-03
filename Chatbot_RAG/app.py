import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


# Input field for the user's Google Gemini API key
api_key = st.text_input("Enter your Google API Key", type="password")

# Extract text from all uploaded PDF files
def get_pdf_text(pdf_docs):
    text = " "
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text = text + page.extract_text()
    return text

# Split extracted text into smaller overlapping chunks
# to improve retrieval accuracy
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

# Generate embeddings and store them in a FAISS vector database
def get_vector_store(text_chunks):

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectore_store = FAISS.from_texts(text_chunks,embedding=embeddings)
    vectore_store.save_local('faiss_index')   # Save the FAISS index locally

# Create the Question Answering chain using Langchain Load qa chain Function

def get_conversational_chain():
    prompt_template = '''
            Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
            provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
            Context:\n {context}?\n
            Question: \n{question}\n

            Answer:
        '''
    model = ChatGoogleGenerativeAI(model='models/gemini-3.6-flash',google_api_key=api_key) # LLM
    prompt = PromptTemplate(template=prompt_template, input_variables=['context','question'])
    chain = load_qa_chain(model,chain_type="stuff",prompt=prompt) # Langchain load qa chain
    return chain

# Retrieve relevant document chunks and generate an answer

def user_input(user_question):

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    new_db = FAISS.load_local('faiss_index',embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({'input_documents': docs, 'question' : user_question})
    st.write('Reply: ', response['output_text'])


# Streamlit Web Interface
def main():
    st.header("AI clone chatbot💁")

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    if user_question and api_key:  # Ensure API key and user question are provided
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button") and pdf_docs:  # Check if API key is provided before processing
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()






