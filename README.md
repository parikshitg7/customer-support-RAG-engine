# Customer Support RAG Engine


[![Live Demo](https://img.shields.io/badge/Demo-Live%20On%20Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)]([YOUR_STREAMLIT_URL_HERE](https://pdf-rag-engine.streamlit.app/))

### [ Click Here to Try the Live App](https://pdf-rag-engine.streamlit.app/)

A lightweight, efficient, and cost-effective **Retrieval-Augmented Generation (RAG)** engine built to perform semantic search and factual question-answering over local documents. This pipeline uses an open-source local embedding model to convert text into mathematical vectors, stores them in an offline vector database, and leverages **Groq's ultra-fast inference infrastructure** to generate grounded, hallucination-resistant answers.

---

##  Features

- **100% Private Semantic Search**  
  Document text is chunked and embedded locally using HuggingFace models—no external embedding API calls required.

- **Hallucination-Resistant Responses**  
  The LLM is constrained via prompt engineering to answer strictly from retrieved document context. If information is unavailable, it safely responds with *"I don't know based on the provided document."*

- **Blazing Fast Answer Generation**  
  Powered by **Groq Cloud API** with Meta's **llama-3.1-8b-instant** model for ultra-low latency responses.

- **Production-Style Architecture**  
  Clean modular separation between:
  - document ingestion
  - chunking
  - embeddings
  - vector database setup
  - retrieval
  - answer generation

- **Fully Local Knowledge Base**  
  ChromaDB stores vectors locally, making the system efficient, portable, and privacy-friendly.

---

##  Tech Stack

### Framework / Orchestration
- LangChain
- LangChain Core
- LangChain Community
- LangChain Groq
- LangChain HuggingFace

### Embedding Model
- HuggingFace `all-MiniLM-L6-v2`
- Local embedding generation
- 384-dimensional vector embeddings

### Vector Database
- ChromaDB (local persistent storage)

### LLM Provider
- Groq Cloud API
- `llama-3.1-8b-instant`

### Utilities
- Python
- python-dotenv
- PyPDF
- sentence-transformers
- venv

---

##  Project Structure

```plaintext
customer-support-RAG-engine/
│
├── chroma_db/                              # Local vector database (ignored from Git)
├── venv/                                   # Virtual environment (ignored from Git)
│
├── Software_Engineering_Complete_Notes.pdf # Knowledge base PDF
│
├── database.py                             # PDF loading, chunking, embeddings, DB creation
├── retrieve.py                             # Semantic retrieval testing
├── rag_engine.py                           # Main RAG pipeline with Groq integration
│
├── .env                                    # API keys (ignored from Git)
├── .gitignore
└── README.md
```

---

##  Setup & Installation

### 1. Clone Repository

```bash
git clone https://github.com/parikshitg7/customer-support-RAG-engine.git
cd customer-support-RAG-engine
```

---

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate:

**Windows (PowerShell)**

```powershell
.\venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install langchain langchain-community langchain-core langchain-huggingface langchain-groq chromadb python-dotenv pypdf sentence-transformers
```

---

##  Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_actual_groq_api_key_here
```

---

##  Usage Guide

### Step 1: Build the Vector Database

This loads the PDF, chunks the content, creates embeddings, and stores them in ChromaDB.

```bash
python database.py
```

Expected flow:

```text
Loading PDF...
Chunking document...
Generating embeddings...
Creating vector database...
Done.
```

---

### Step 2: Test Semantic Retrieval (Optional)

Run retrieval testing before connecting the LLM:

```bash
python retrieve.py
```

This verifies semantic search is returning relevant chunks.

---

### Step 3: Run the RAG Engine

Launch the full pipeline:

```bash
python rag_engine.py
```

System flow:

```text
User Query
   ↓
Semantic Retrieval from ChromaDB
   ↓
Top Matching Chunks
   ↓
Prompt Assembly
   ↓
Groq LLM
   ↓
Grounded Answer
```

---

##  Example Query

```text
What is Retrieval-Augmented Generation?
```

Example response:

```text
Retrieval-Augmented Generation (RAG) combines information retrieval with language generation by first searching relevant document context and then generating an answer grounded in that information.
```

---

##  Core Architecture

```text
PDF Document
   ↓
PyPDF Loader
   ↓
Document Pages
   ↓
Recursive Character Text Splitter
   ↓
Chunks
   ↓
HuggingFace Embeddings
   ↓
ChromaDB Vector Store
   ↓
Semantic Retrieval
   ↓
Groq LLM
   ↓
Final Answer
```

---

## 🔒 Privacy Notes

This project keeps document processing local:

 PDF parsing is local  
 Chunking is local  
 Embeddings are generated locally  
 Vector DB is local  

Only the final retrieved context is sent to Groq for answer generation.

---

##  Future Improvements

- FastAPI REST API wrapper
- Streamlit frontend UI
- Multi-document upload support
- Chat memory support
- Source citation display
- Hybrid search (keyword + semantic)
- Docker containerization
- Cloud deployment

---

## 📄 License

This project is built for educational and portfolio purposes.