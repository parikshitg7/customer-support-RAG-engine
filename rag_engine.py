import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq 

# --- THE FIX IS HERE: Using langchain_classic ---
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# ------------------------------------------------

from langchain_core.prompts import ChatPromptTemplate

def run_rag_assistant():
    print("--- Starting Phase 5: The RAG Assistant (Powered by Groq) ---")
    
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: Please add your GROQ_API_KEY to the .env file!")
        return

    print("Connecting to Vector DB...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
    
    print("Waking up Llama 3 on Groq...")
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
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    question = "What are the main goals of Software Engineering?"
    print(f"\nUser Question: '{question}'")
    print("Thinking (at lightning speed)...\n")
    
    response = rag_chain.invoke({"input": question})
    
    print("=== AI ANSWER ===")
    print(response["answer"])

if __name__ == "__main__":
    run_rag_assistant()