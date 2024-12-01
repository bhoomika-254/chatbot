import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import io

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PDF_PATH = r"C:\Users\bhoom\OneDrive\Desktop\isro project\chatbot\PavanAI.pdf"

# Domain-specific keywords
DOMAIN_KEYWORDS = [
    "AQI", "Air Quality Index", "pollution", "environmental", "monitoring", 
    "air quality", "sensor", "data", "measurement", "atmospheric"
]

def validate_domain_relevance(question):
    """Check if the question is related to the project domain"""
    question_lower = question.lower()
    return any(keyword.lower() in question_lower for keyword in DOMAIN_KEYWORDS)

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

def handle_image_input(uploaded_file):
    try:
        # Configure Gemini
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # Read file contents
        file_bytes = uploaded_file.getvalue()
        
        # Create a file object with mime type
        file = genai.upload_file(
            path=io.BytesIO(file_bytes), 
            mime_type=uploaded_file.type
        )
        
        # Use Gemini Pro Vision model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Generate content description
        response = model.generate_content([file, "Describe this image in detail"])
        return response.text
    except Exception as e:
        st.error(f"Image processing error: {e}")
        return "Unable to analyze the image."

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not precisely in the context, indicate limited information.
    Do not fabricate answers.

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

def get_gemini_response(user_question, is_domain_relevant=False):
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
        
        if not is_domain_relevant:
            return "I do not have expertise in this domain."
        
        chat = model.start_chat(history=[])
        response = chat.send_message(user_question)
        return response.text
    except Exception as e:
        st.error(f"Error generating Gemini response: {e}")
        return "Error in generating response from Gemini."

def main():
    # Page configuration
    st.set_page_config(
        page_title="Pavan AI - AQI Insights",
        page_icon="🌍",
        layout="centered"
    )

    # Custom CSS
    st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #f6f8f9 0%, #e5ebee 100%);
        }
        .stApp {
            background-color: rgba(255,255,255,0.9);
            border-radius: 15px;
            padding: 20px;
            max-width: 800px;
            margin: auto;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .chat-message {
            padding: 15px;
            margin: 10px 0;
            border-radius: 12px;
            max-width: 80%;
        }
        .user-message {
            background-color: #E6F3FF;
            color: #003366;
            align-self: flex-end;
            margin-left: auto;
        }
        .assistant-message {
            background-color: #F0F8FF;
            color: #00254D;
            align-self: flex-start;
            margin-right: auto;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("🌍 Pavan AI - AQI Insights")
    st.caption("Your AI Companion for Air Quality Intelligence")

    # Process PDF during startup
    pdf_text = get_pdf_text(PDF_PATH)
    if pdf_text:
        text_chunks = get_text_chunks(pdf_text)
        get_vector_store(text_chunks)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Image upload
    uploaded_file = st.file_uploader("Upload an image (Optional)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=250)
        image_analysis = handle_image_input(uploaded_file)
        st.write("Image Analysis:", image_analysis)

    # Chat input
    user_question = st.chat_input("Ask about Air Quality, AQI, or related topics")

    if user_question:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        # Check context in PDF
        context_response = user_input_handler(user_question)
        
        # Check domain relevance
        is_domain_relevant = validate_domain_relevance(user_question)
        
        # Determine final response
        if context_response != "The answer is not available in the context.":
            response = context_response
        else:
            response = get_gemini_response(user_question, is_domain_relevant)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Display chat history
    for message in st.session_state.messages:
        role_class = "user-message" if message["role"] == "user" else "assistant-message"
        st.markdown(f"<div class='chat-message {role_class}'>{message['content']}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()