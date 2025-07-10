
# AI Compliance Co-Pilot

## Introduction

Compliance professionals spend countless hours sifting through dense regulatory documents, struggling to find relevant information buried in hundreds of pages of legal text. The AI Compliance Co-Pilot transforms this tedious process into an intelligent conversation. Our AI-powered framework uses advanced Retrieval-Augmented Generation (RAG) technology to provide instant, citable answers to compliance questions, turning complex regulatory documents into an accessible knowledge base that speaks your language.

## Core Features

• **Intelligent Search** - Natural language querying that understands context and intent, not just keywords
• **Centralized Knowledge Base** - Unified access to all your compliance documents in one searchable repository  
• **On-Demand Summarization** - Instant extraction of key insights with direct citations to source documents

## Tech Stack

- **Python** - Core programming language
- **FastAPI** - High-performance REST API framework
- **Streamlit** - Interactive web application frontend
- **LangChain** - Orchestration framework for LLM applications
- **Google Gemini** - Advanced language model for answer generation
- **ChromaDB** - Vector database for semantic search
- **Unstructured** - Document processing and chunking library

## Architecture

The system uses the Unstructured library to ingest and chunk documents. LangChain orchestrates a RAG pipeline where user queries retrieve relevant text chunks from a ChromaDB vector store, which are then passed with the query to Google's Gemini Pro model for final answer generation.

```
Documents → Unstructured → Text Chunks → ChromaDB → Similarity Search → Context + Query → Gemini Pro → Answer
```

## Setup and Installation

Follow these steps to run the AI Compliance Co-Pilot locally:

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-compliance-co-pilot.git
cd ai-compliance-co-pilot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root and add your Google API key:

```bash
GOOGLE_API_KEY=your_google_api_key_here
```

### 4. Prepare Your Documents

Place your compliance documents in the `./data` directory. Supported formats include:
- PDF (.pdf)
- Word Documents (.docx)
- Text Files (.txt)
- Markdown (.md)
- HTML (.html)

### 5. Run the Document Ingestion

Process your documents and create the vector database:

```bash
python ingest.py
```

### 6. Start the Backend Server

Launch the FastAPI backend server:

```bash
uvicorn backend.main:app --reload
```

The API will be available at `http://localhost:8000`

### 7. Launch the Frontend

In a new terminal window, start the Streamlit frontend:

```bash
streamlit run app.py
```

The web application will open in your browser at `http://localhost:8501`

## Usage

1. **Upload Documents**: Place your compliance documents in the `./data` folder
2. **Process Documents**: Run the ingestion script to create the knowledge base
3. **Ask Questions**: Use natural language to query your documents through the web interface
4. **Get Instant Answers**: Receive contextual answers with source citations

## API Documentation

Once the backend is running, visit `http://localhost:8000/docs` for interactive API documentation.

## About This Project

This is a college project developed to demonstrate the application of AI and machine learning technologies in the compliance domain. The project showcases the integration of various modern technologies including RAG pipelines, vector databases, and large language models.

---

*An AI-powered solution for intelligent document processing and compliance automation.*