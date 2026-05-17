import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq 
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Setup the Page
st.set_page_config(page_title="RAG Engine", page_icon="🤖")
st.title("📚 Software Engineering Assistant")
st.caption("Powered by local ChromaDB and Llama 3.1 on Groq")

# 2. Cache the heavy lifting so it doesn't reload on every single click
@st.cache_resource
def load_rag_chain():
    load_dotenv()
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    system_prompt = (
        "You are a strict, helpful assistant for a Software Engineering class. "
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
    return create_retrieval_chain(retriever, question_answer_chain)

# Boot up the RAG engine in the background
rag_chain = load_rag_chain()

# 3. Setup Chat Memory
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me anything about the Software Engineering document."}]

# Display all previous chat messages on the screen
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 4. The Input Box
if user_input := st.chat_input("Ask your question here..."):
    # Show user message instantly
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    
    # Get AI response with a cool loading spinner
    with st.spinner("Searching document & thinking..."):
        response = rag_chain.invoke({"input": user_input})
        answer = response["answer"]
        
    # Show AI response
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)