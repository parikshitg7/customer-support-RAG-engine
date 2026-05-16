from langchain_community.document_loaders import PyPDFLoader

def load_pdf():
    print("--- Starting PDF Ingestion ---")
    
    # 1. Point to your PDF file
    file_path = "Software_Engineering_Complete_Notes.pdf"
    
    # 2. Initialize the PyPDFLoader
    loader = PyPDFLoader(file_path)
    
    # 3. Load the document and split it into pages
    pages = loader.load()
    
    # 4. Check what we got
    print(f"Successfully loaded {len(pages)} pages!")
    
    # Let's inspect the first page to see what it looks like
    #for doc in pages:
        #print(doc.page_content)
    if len(pages) > 0:
        first_page = pages[0]
        print("\n--- Metadata of Page 1 ---")
        print(first_page.metadata)
        
        print("\n--- First 300 Characters of Page 1 Content ---")
        print(first_page.page_content[:300])

if __name__ == "__main__":
    load_pdf()