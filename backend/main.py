#!/usr/bin/env python3
"""
FastAPI RAG Backend - main.py

A production-ready FastAPI application implementing Retrieval-Augmented Generation (RAG)
using ChromaDB vector store and Google Generative AI.

Author: Senior Python Backend Developer
Date: July 2025
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# LangChain imports
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Environment configuration
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for storing initialized components
app_state = {
    "vector_store": None,
    "embeddings": None,
    "llm": None,
    "rag_chain": None
}

# Pydantic models for request/response
class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    max_results: Optional[int] = Field(default=5, ge=1, le=20, description="Maximum number of results to retrieve")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty or only whitespace')
        return v.strip()

class SearchResponse(BaseModel):
    """Response model for search endpoint."""
    answer: str = Field(..., description="Generated answer from RAG system")
    sources: Optional[List[str]] = Field(default=None, description="Source documents used for the answer")
    query: str = Field(..., description="Original query")

class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    message: str
    components: Dict[str, str]

# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown events."""
    logger.info("Starting FastAPI RAG application...")
    
    try:
        # Load environment variables from parent directory
        env_path = Path(__file__).parent.parent / ".env"
        load_dotenv(env_path)
        logger.info(f"✓ Environment loaded from {env_path}")
        
        # Initialize components
        await initialize_components()
        logger.info("✓ All components initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise
    finally:
        logger.info("Shutting down FastAPI RAG application...")

# Initialize FastAPI app
app = FastAPI(
    title="RAG Search API",
    description="A Retrieval-Augmented Generation API using ChromaDB and Google Generative AI",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

async def initialize_components():
    """Initialize all RAG system components."""
    
    # Validate environment variables
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    logger.info("✓ Google API key loaded")
    
    # Initialize embeddings
    try:
        app_state["embeddings"] = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        logger.info("✓ Google Generative AI embeddings initialized")
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {str(e)}")
        raise
    
    # Load ChromaDB vector store
    chroma_db_path = Path(__file__).parent.parent / "chroma_db"
    
    if not chroma_db_path.exists():
        raise FileNotFoundError(f"ChromaDB vector store not found at {chroma_db_path}")
    
    try:
        app_state["vector_store"] = Chroma(
            persist_directory=str(chroma_db_path),
            embedding_function=app_state["embeddings"]
        )
        logger.info(f"✓ ChromaDB vector store loaded from {chroma_db_path}")
    except Exception as e:
        logger.error(f"Failed to load ChromaDB: {str(e)}")
        raise
    
    # Initialize LLM
    try:
        app_state["llm"] = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.7,
            convert_system_message_to_human=True
        )
        logger.info("✓ ChatGoogleGenerativeAI (gemini-1.5-flash) initialized")
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}")
        raise
    
    # Setup RAG chain using LangChain Expression Language (LCEL)
    try:
        # Create prompt template
        prompt_template = ChatPromptTemplate.from_template(
            """You are a helpful AI assistant that answers questions based on the provided context.
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer based on the context, just say that you don't know.
            Don't try to make up an answer.

            Context: {context}

            Question: {question}

            Answer: """
        )
        
        # Create retriever
        retriever = app_state["vector_store"].as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Setup RAG chain using LCEL
        app_state["rag_chain"] = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt_template
            | app_state["llm"]
            | StrOutputParser()
        )
        
        logger.info("✓ RAG chain initialized using LCEL")
        
    except Exception as e:
        logger.error(f"Failed to setup RAG chain: {str(e)}")
        raise

def format_docs(docs: List[Document]) -> str:
    """Format documents for context in prompt."""
    return "\n\n".join([doc.page_content for doc in docs])

def get_rag_chain():
    """Dependency to get the RAG chain."""
    if app_state["rag_chain"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized"
        )
    return app_state["rag_chain"]

def get_vector_store():
    """Dependency to get the vector store."""
    if app_state["vector_store"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store not initialized"
        )
    return app_state["vector_store"]

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Search API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    components = {
        "vector_store": "✓ Ready" if app_state["vector_store"] else "✗ Not initialized",
        "embeddings": "✓ Ready" if app_state["embeddings"] else "✗ Not initialized",
        "llm": "✓ Ready" if app_state["llm"] else "✗ Not initialized",
        "rag_chain": "✓ Ready" if app_state["rag_chain"] else "✗ Not initialized"
    }
    
    all_ready = all("✓" in status for status in components.values())
    
    return HealthResponse(
        status="healthy" if all_ready else "unhealthy",
        message="All components ready" if all_ready else "Some components not ready",
        components=components
    )

@app.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    rag_chain=Depends(get_rag_chain),
    vector_store=Depends(get_vector_store)
):
    """
    Search endpoint that performs RAG-based question answering.
    
    Args:
        request: Search request containing query and optional parameters
        
    Returns:
        SearchResponse containing the generated answer and source information
    """
    try:
        logger.info(f"Processing search query: {request.query}")
        
        # Invoke the RAG chain
        answer = await rag_chain.ainvoke(request.query)
        
        # Get source documents for transparency
        try:
            source_docs = vector_store.similarity_search(
                request.query, 
                k=request.max_results
            )
            sources = [
                doc.metadata.get("source_file", "Unknown source") 
                for doc in source_docs
            ]
            # Remove duplicates while preserving order
            sources = list(dict.fromkeys(sources))
        except Exception as e:
            logger.warning(f"Failed to retrieve source documents: {str(e)}")
            sources = None
        
        logger.info(f"Successfully generated answer for query: {request.query}")
        
        return SearchResponse(
            answer=answer,
            sources=sources,
            query=request.query
        )
        
    except Exception as e:
        logger.error(f"Error processing search query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process search query: {str(e)}"
        )

@app.get("/stats", response_model=Dict[str, Any])
async def get_stats(vector_store=Depends(get_vector_store)):
    """Get statistics about the vector store."""
    try:
        # Get collection info
        collection = vector_store._collection
        count = collection.count()
        
        return {
            "total_documents": count,
            "vector_store_path": str(Path(__file__).parent.parent / "chroma_db"),
            "embedding_model": "models/embedding-001",
            "llm_model": "gemini-1.5-flash"
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unhandled exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )

if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )