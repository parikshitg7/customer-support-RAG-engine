from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_document():
    print("--- Starting Phase 2: Chunking ---")
    
    # 1. Load the document (Phase 1 logic)
    loader = PyPDFLoader("Software_Engineering_Complete_Notes.pdf")
    pages = loader.load()
    
    # 2. Initialize the Text Splitter
    # We use RecursiveCharacterTextSplitter because it smartly splits by paragraphs, 
    # then sentences, then words, keeping structural meaning intact.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # Max characters per chunk
        chunk_overlap=50,     # Overlap between chunks
        length_function=len
    )
    
    # 3. Split the pages into chunks
    chunks = text_splitter.split_documents(pages)
    
    # 4. Inspect the results
    print(f"Total pages loaded: {len(pages)}")
    print(f"Total chunks created: {len(chunks)}")
    
    # Let's look closely at Chunk 0 and Chunk 1 to see the overlap
    # Let's look closely at Chunk 1 and Chunk 2 to see the overlap
    if len(chunks) > 2:
        print("\n=== CHUNK 1 ===")
        print(chunks[1].page_content)
        
        print("\n=== CHUNK 2 ===")
        print(chunks[2].page_content)

if __name__ == "__main__":
    chunk_document()