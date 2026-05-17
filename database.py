from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def build_database():
    print("--- Starting Phase 3: Building Vector DB ---")
    
    # 1. Load and Chunk (Repeating Phases 1 & 2)
    print("Loading and chunking PDF...")
    loader = PyPDFLoader("Software_Engineering_Complete_Notes.pdf") # Update name if your file is named differently
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    chunks = text_splitter.split_documents(pages)
    print(f"Prepared {len(chunks)} chunks.")

    # 2. Initialize the Embedding Model
    # This downloads a small, highly efficient open-source model the first time you run it.
    print("\nInitializing Embedding Model (this might take a moment to download on first run)...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 3. Create and Save to ChromaDB
    print("\nConverting chunks to math and saving to ChromaDB...")
    db_path = "./chroma_db"
    
    # This does the heavy lifting: embeds the text and saves it to a folder
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model, 
        persist_directory=db_path
    )
    
    print(f"\nSuccess! Database saved to the '{db_path}' folder.")

if __name__ == "__main__":
    build_database()