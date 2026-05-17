from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def search_database():
    print("--- Starting Phase 4: The Retriever ---")
    
    # 1. Initialize the EXACT SAME Embedding Model
    # The question must be translated using the exact same dictionary as the answers!
    print("Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Connect to the existing Chroma Database
    print("Connecting to local ChromaDB...")
    db_path = "./chroma_db"
    vector_db = Chroma(persist_directory=db_path, embedding_function=embedding_model)
    
    # 3. Define the User's Question
    # Since your PDF is about Software Engineering, let's ask a relevant question
    query = "What are the main goals of Software Engineering?"
    print(f"\nUser Question: '{query}'")
    
    # 4. Perform the Similarity Search
    # We ask the DB to return the top 3 most mathematically similar chunks (k=3)
    print("Searching for the 3 most relevant chunks...\n")
    results = vector_db.similarity_search(query, k=3)
    
    # 5. Display the Results
    for i, doc in enumerate(results):
        print(f"=== RESULT {i + 1} ===")
        print(f"Found on Page: {doc.metadata.get('page', 'Unknown')}")
        print(f"Text: {doc.page_content}\n")

if __name__ == "__main__":
    search_database()