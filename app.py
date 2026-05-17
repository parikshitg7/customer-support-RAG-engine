import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# New imports for document processing
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq 
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Setup the Page
st.set_page_config(page_title="Dynamic RAG Engine", page_icon="🤖")
st.title("📚 Chat with ANY PDF")
st.caption("Upload a document and ask questions powered by Llama 3.1")

# Load environment variables
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    st.error("Missing GROQ_API_KEY. Please add it to your secrets or .env file.")
    st.stop()

# 2. Sidebar for File Upload
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload your PDF here", type=["pdf"])
    st.markdown("---")
    st.markdown("### How it works:")
    st.markdown("1. Upload a PDF.\n2. The system reads and chunks the text.\n3. It builds a temporary Vector Database.\n4. You can ask questions!")

# 3. Process the Uploaded PDF
# We use @st.cache_resource so it doesn't re-process the PDF every time you type a question
@st.cache_resource
def process_document(file_to_process):
    with st.spinner("Reading, chunking, and embedding document... This takes a few seconds."):
        # Streamlit uploads files to memory. PyPDFLoader needs a real file path.
        # So, we create a temporary file on the hard drive to trick it.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_to_process.getvalue())
            temp_filepath = temp_file.name

        # Load and Chunk the PDF
        loader = PyPDFLoader(temp_filepath)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        # Build an IN-MEMORY vector database (no persist_directory)
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = Chroma.from_documents(documents=chunks, embedding=embedding_model)

        # Initialize the LLM
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
        
        # Clean up the temp file
        os.remove(temp_filepath)
        
        return rag_chain

# 4. Chat Interface Logic
# ONLY show the chat if a file has been uploaded and processed
if uploaded_file is not None:
    # Process the file and get the RAG chain
    rag_chain = process_document(uploaded_file)
    
    st.success("✅ Document processed successfully! Ask your questions below.")

    # Setup Chat Memory
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Document loaded! What would you like to know?"}]

    # Display all previous chat messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # The Input Box
    if user_input := st.chat_input("Ask a question about your PDF..."):
        # Show user message instantly
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        
        # Get AI response
        with st.spinner("Searching document..."):
            response = rag_chain.invoke({"input": user_input})
            answer = response["answer"]
            
        # Show AI response
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
else:
    # What to show when no file is uploaded yet
    st.info("👈 Please upload a PDF in the sidebar to get started.")