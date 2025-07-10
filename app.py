#!/usr/bin/env python3
"""
Streamlit RAG Frontend - app.py

A user-friendly Streamlit web application for interacting with the RAG system.
Provides a clean interface for document search and question answering.

Author: Python UI Developer
Date: July 2025
"""

import streamlit as st
import requests
import json
import time
from typing import Dict, Any, Optional

# Configure page settings
st.set_page_config(
    page_title="RAG Search Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_BASE_URL = "http://127.0.0.1:8000"
SEARCH_ENDPOINT = f"{API_BASE_URL}/search"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
STATS_ENDPOINT = f"{API_BASE_URL}/stats"

def check_backend_health() -> bool:
    """Check if the backend server is running and healthy."""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def get_backend_stats() -> Optional[Dict[str, Any]]:
    """Get statistics from the backend server."""
    try:
        response = requests.get(STATS_ENDPOINT, timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        pass
    return None

def search_documents(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Send search query to the backend API.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to retrieve
        
    Returns:
        Dictionary containing the API response or error information
    """
    try:
        payload = {
            "query": query,
            "max_results": max_results
        }
        
        response = requests.post(
            SEARCH_ENDPOINT,
            json=payload,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return {
                "success": True,
                "data": response.json()
            }
        else:
            return {
                "success": False,
                "error": f"API returned status code {response.status_code}: {response.text}"
            }
            
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Could not connect to the backend server."
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timed out. Please try again."
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Request failed: {str(e)}"
        }

def display_answer(answer: str, sources: Optional[list] = None, query: str = ""):
    """Display the search results in a formatted way."""
    
    # Display the answer
    st.markdown("### 📝 Answer")
    st.markdown(answer)
    
    # Display sources if available
    if sources:
        st.markdown("### 📚 Sources")
        with st.expander("View source documents", expanded=False):
            for i, source in enumerate(sources, 1):
                st.markdown(f"**{i}.** {source}")
    
    # Display query for reference
    if query:
        st.markdown("### 🔍 Your Query")
        st.info(f"**Query:** {query}")

def main():
    """Main application function."""
    
    # Page header
    st.title("🔍 RAG Search Assistant")
    st.markdown("Ask questions about your documents and get AI-powered answers!")
    
    # Sidebar for system status and settings
    with st.sidebar:
        st.header("🛠️ System Status")
        
        # Check backend health
        if check_backend_health():
            st.success("✅ Backend server is running")
            
            # Get and display stats
            stats = get_backend_stats()
            if stats:
                st.markdown("### 📊 Statistics")
                st.metric("Total Documents", stats.get("total_documents", "Unknown"))
                st.markdown(f"**Embedding Model:** {stats.get('embedding_model', 'Unknown')}")
                st.markdown(f"**LLM Model:** {stats.get('llm_model', 'Unknown')}")
        else:
            st.error("❌ Backend server is not responding")
            st.markdown("Please ensure the FastAPI server is running on http://127.0.0.1:8000")
        
        st.divider()
        
        # Settings
        st.header("⚙️ Settings")
        max_results = st.slider(
            "Maximum results to retrieve",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of relevant documents to retrieve for context"
        )
        
        st.divider()
        
        # Instructions
        st.header("💡 How to Use")
        st.markdown("""
        1. Enter your question in the text input
        2. Click 'Search' or press Enter
        3. Wait for the AI to process your query
        4. Review the answer and sources
        """)
        
        st.markdown("### 🔥 Example Queries")
        st.markdown("""
        - "What are the main topics covered?"
        - "Summarize the key findings"
        - "What are the recommendations?"
        - "Explain the methodology used"
        """)
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Query input
        query = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know about your documents?",
            help="Type your question and press Enter or click Search"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Process search when button is clicked or Enter is pressed
    if search_button or (query and query.strip()):
        if not query or not query.strip():
            st.warning("⚠️ Please enter a question to search.")
            return
        
        # Check if backend is available
        if not check_backend_health():
            st.error("❌ Error: Could not connect to the backend server.")
            st.markdown("Please ensure the FastAPI server is running on http://127.0.0.1:8000")
            return
        
        # Perform search with spinner
        with st.spinner("Searching..."):
            result = search_documents(query.strip(), max_results)
        
        # Display results
        if result["success"]:
            data = result["data"]
            display_answer(
                answer=data["answer"],
                sources=data.get("sources"),
                query=data["query"]
            )
            
            # Add some spacing and a success message
            st.success("✅ Search completed successfully!")
            
        else:
            st.error(f"❌ Error: {result['error']}")
            
            # Provide troubleshooting tips
            with st.expander("💡 Troubleshooting Tips"):
                st.markdown("""
                **Common issues and solutions:**
                
                1. **Backend not running**: Start the FastAPI server with `python backend/main.py`
                2. **Port conflicts**: Ensure port 8000 is not being used by another application
                3. **Network issues**: Check your internet connection and firewall settings
                4. **API errors**: Check the backend logs for detailed error messages
                """)
    
    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>🚀 Powered by LangChain, ChromaDB, and Google Generative AI</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()