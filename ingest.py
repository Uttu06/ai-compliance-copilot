#!/usr/bin/env python3
"""
Document Processing Pipeline - ingest.py

A production-ready script for ingesting documents into a ChromaDB vector store
using LangChain and Google Generative AI embeddings.

Author: Lead Python Engineer
Date: July 2025
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional

# LangChain imports
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Environment configuration
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingest.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Document processing pipeline for ChromaDB vector store creation."""
    
    def __init__(self, 
                 chroma_db_path: str = "./chroma_db",
                 source_data_path: str = "./data",
                 chunk_size: int = 2000,
                 chunk_overlap: int = 400):
        """
        Initialize the document processor.
        
        Args:
            chroma_db_path: Path to persist ChromaDB vector store
            source_data_path: Path to source documents directory
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chroma_db_path = Path(chroma_db_path)
        self.source_data_path = Path(source_data_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.text_splitter = None
        self.embeddings = None
        self.documents = []
        self.chunks = []
        
    def load_environment(self) -> None:
        """Load environment variables and validate API key."""
        logger.info("Loading environment variables...")
        load_dotenv()
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in environment variables. "
                "Please set it in your .env file or environment."
            )
        logger.info("✓ Google API key loaded successfully")
        
    def validate_paths(self) -> None:
        """Validate source and destination paths."""
        logger.info("Validating paths...")
        
        # Check source data directory exists
        if not self.source_data_path.exists():
            raise FileNotFoundError(f"Source data directory not found: {self.source_data_path}")
        
        # Create chroma_db directory if it doesn't exist
        self.chroma_db_path.mkdir(parents=True, exist_ok=True)
        
        # Check if source directory has files
        files = list(self.source_data_path.glob("*"))
        if not files:
            raise ValueError(f"No files found in source directory: {self.source_data_path}")
        
        logger.info(f"✓ Found {len(files)} files in source directory")
        
    def load_documents(self) -> None:
        """Load all documents from the source directory."""
        logger.info("Loading documents...")
        
        supported_extensions = {'.txt', '.pdf', '.docx', '.md', '.html', '.csv', '.json'}
        loaded_count = 0
        
        # Get all files from source directory
        for file_path in self.source_data_path.iterdir():
            if file_path.is_file():
                # Check if file extension is supported
                if file_path.suffix.lower() in supported_extensions:
                    try:
                        loader = UnstructuredFileLoader(str(file_path))
                        docs = loader.load()
                        
                        # Add metadata to documents
                        for doc in docs:
                            doc.metadata.update({
                                'source_file': str(file_path),
                                'file_name': file_path.name,
                                'file_extension': file_path.suffix
                            })
                        
                        self.documents.extend(docs)
                        loaded_count += 1
                        logger.info(f"✓ Loaded: {file_path.name}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path.name}: {str(e)}")
                        continue
                else:
                    logger.warning(f"Skipping unsupported file type: {file_path.name}")
        
        if not self.documents:
            raise ValueError("No documents were successfully loaded")
        
        logger.info(f"✓ Successfully loaded {loaded_count} documents with {len(self.documents)} document chunks")
        
    def split_documents(self) -> None:
        """Split documents into chunks using RecursiveCharacterTextSplitter."""
        logger.info("Splitting documents into chunks...")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Split documents into chunks
        self.chunks = self.text_splitter.split_documents(self.documents)
        
        # Print total number of chunks created
        chunk_count = len(self.chunks)
        print(f"Created {chunk_count} text chunks.")
        logger.info(f"✓ Created {chunk_count} text chunks")
        
        if chunk_count == 0:
            raise ValueError("No text chunks were created from the documents")
            
    def initialize_embeddings(self) -> None:
        """Initialize Google Generative AI embeddings."""
        logger.info("Initializing Google Generative AI embeddings...")
        
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001"
            )
            logger.info("✓ Google Generative AI embeddings initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            raise
            
    def create_vector_store(self) -> None:
        """Create and persist ChromaDB vector store."""
        logger.info("Creating ChromaDB vector store...")
        
        try:
            # Remove existing database if it exists
            if self.chroma_db_path.exists():
                import shutil
                shutil.rmtree(self.chroma_db_path)
                logger.info("✓ Removed existing vector store")
            
            # Create new vector store
            vector_store = Chroma.from_documents(
                documents=self.chunks,
                embedding=self.embeddings,
                persist_directory=str(self.chroma_db_path)
            )
            
            # Persist the vector store
            vector_store.persist()
            logger.info(f"✓ Vector store persisted to {self.chroma_db_path}")
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {str(e)}")
            raise
            
    def run_pipeline(self) -> None:
        """Execute the complete document processing pipeline."""
        try:
            logger.info("Starting document processing pipeline...")
            
            # Step 1: Load environment
            self.load_environment()
            
            # Step 2: Validate paths
            self.validate_paths()
            
            # Step 3: Load documents
            self.load_documents()
            
            # Step 4: Split documents
            self.split_documents()
            
            # Step 5: Initialize embeddings
            self.initialize_embeddings()
            
            # Step 6: Create vector store
            self.create_vector_store()
            
            # Success message
            print("Vector store created successfully.")
            logger.info("✓ Document processing pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    """Main entry point for the document processing pipeline."""
    try:
        # Initialize processor with default configuration
        processor = DocumentProcessor(
            chroma_db_path="./chroma_db",
            source_data_path="./data",
            chunk_size=2000,
            chunk_overlap=400
        )
        
        # Run the pipeline
        processor.run_pipeline()
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()