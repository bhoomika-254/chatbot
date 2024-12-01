import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables

st.set_page_config(
    page_title="Pavan AI",
    page_icon=favicon_bytes or "🤖",  # Use the custom favicon if available
    layout="centered",
    initial_sidebar_state="collapsed",
)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PDF_PATH = r"C:\Users\bhoom\OneDrive\Desktop\isro project\chatbot\Gemini_API_Chatbot_using_streamlit\PavanAI.pdf"
GOOGLE_CREDENTIALS_PATH = r"C:\Users\bhoom\OneDrive\Desktop\isro project\chatbot\Gemini_API_Chatbot_using_streamlit\chatbot-service-account-442504-6c01ffe9905f.json"

# Validate Google API Key and Credentials Path
if not GOOGLE_API_KEY or not os.path.exists(GOOGLE_CREDENTIALS_PATH):
    st.error("Google API Key or Credentials file is missing! Check your environment variables.")
else:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIALS_PATH
    genai.configure(api_key=GOOGLE_API_KEY)

# Utility functions (unchanged)
def get_pdf_text(pdf_path):
    try:
        pdf_reader = PdfReader(pdf_path)
        return "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
    except FileNotFoundError:
        st.error("PDF file not found! Please check the path.")
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return ""

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error creating FAISS vector store: {e}")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, say: "The answer is not available in the context."
    Do not provide incorrect or fabricated answers.

    Context:\n{context}\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input_handler(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return "Error processing your query."
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def response_generator(user_question):
    response = user_input_handler(user_question)
    for word in response.split():
        yield word + " "

# Improved UI
favicon_path = r"C:\Users\bhoom\OneDrive\Desktop\isro project\chatbot\Gemini_API_Chatbot_using_streamlit\favicon.ico"
favicon_bytes = None
if os.path.exists(favicon_path):
    with open(favicon_path, "rb") as f:
        favicon_bytes = f.read()

# CSS Styling for background and chat bubble aesthetics
st.markdown("""
    <style>
        /* Background image styling */
        body {
            background: url('https://www.transparenttextures.com/patterns/scribble-light.png'), linear-gradient(135deg, #f6d365, #fda085);
            background-size: cover;
            color: #2c2c2c;
        }
        /* Chat message bubbles */
        .chat-message {
            padding: 15px;
            margin: 10px 0;
            border-radius: 10px;
        }
        .chat-user {
            background-color: #d6eaf8;
            color: #2c3e50;
            text-align: left;
        }
        .chat-assistant {
            background-color: #d5f5e3;
            color: #145a32;
            text-align: right;
        }
        /* Centered container */
        .streamlit-container {
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Headers and captions
st.header(":sparkles: Pavan AI")
st.caption("Your AI Companion for Smart Conversations related to AQI | By Team Siriius")

# Process PDF
pdf_text = get_pdf_text(PDF_PATH)
if pdf_text:
    text_chunks = get_text_chunks(pdf_text)
    get_vector_store(text_chunks)

# Manage chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    role = "chat-user" if message["role"] == "user" else "chat-assistant"
    st.markdown(f"<div class='chat-message {role}'>{message['content']}</div>", unsafe_allow_html=True)

# Input handler
if prompt := st.chat_input("Ask me anything!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"<div class='chat-message chat-user'>{prompt}</div>", unsafe_allow_html=True)
    response = "".join(response_generator(prompt))
    st.markdown(f"<div class='chat-message chat-assistant'>{response}</div>", unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": response})
