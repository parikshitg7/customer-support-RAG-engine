import os
import tempfile
import uuid
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq 

# Updated imports to langchain_classic to fix the ModuleNotFoundError
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Setup the Page
st.set_page_config(page_title="Dynamic RAG Engine", page_icon="🤖")
st.title("📚 Chat with ANY PDF")
st.caption("Upload a document and ask questions powered by Llama 3.1")

load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    st.error("Missing GROQ_API_KEY. Please add it to your secrets or .env file.")
    st.stop()

# 2. Sidebar for File Upload & Garbage Collection
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload your PDF here", type=["pdf"])
    
    # NEW: The "Shredder" button to prevent memory leaks and reset the app
    if st.button("🗑️ Clear Data & Start Over"):
        # Explicitly delete the Chroma bucket from the server's RAM
        if "vector_db" in st.session_state:
            st.session_state.vector_db.delete_collection()
            
        # Wipe the user's browser session clean
        st.session_state.clear()
        
        # Refresh the page
        st.rerun()

    st.markdown("---")
    st.markdown("### How it works:")
    st.markdown("1. Upload a PDF.\n2. The system reads and chunks the text.\n3. It builds a temporary Vector Database.\n4. You can ask questions!")

# 3. Process the Uploaded PDF securely
def process_document(file_to_process):
    with st.spinner("Reading, chunking, and embedding document... This takes a few seconds."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_to_process.getvalue())
            temp_filepath = temp_file.name

        loader = PyPDFLoader(temp_filepath)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Generate a unique ID to ensure buckets never mix between users
        session_id = str(uuid.uuid4())
        
        vector_db = Chroma.from_documents(
            documents=chunks, 
            embedding=embedding_model,
            collection_name=session_id
        )
        
        # Save the database reference so the Clear Data button can find and delete it
        st.session_state.vector_db = vector_db

        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        
        system_prompt = (
            "You are a strict, helpful assistant. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer or if the answer is not in the context, "
            "just say 'I don't know, this is not covered in the document.' "
            "Do not make up an answer. Keep the answer concise.\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        os.remove(temp_filepath)
        return rag_chain

# 4. Chat Interface Logic
if uploaded_file is not None:
    
    # Store the database in the user's private session_state
    if "processed_file_id" not in st.session_state or st.session_state.processed_file_id != uploaded_file.file_id:
        st.session_state.rag_chain = process_document(uploaded_file)
        st.session_state.processed_file_id = uploaded_file.file_id
        st.session_state.messages = [{"role": "assistant", "content": "Document loaded! What would you like to know?"}]
        
    st.success("✅ Document processed successfully! Ask your questions below.")

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if user_input := st.chat_input("Ask a question about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        
        with st.spinner("Searching document..."):
            response = st.session_state.rag_chain.invoke({"input": user_input})
            answer = response["answer"]
            
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
else:
    st.info("👈 Please upload a PDF in the sidebar to get started.")