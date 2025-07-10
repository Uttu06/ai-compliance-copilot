#!/usr/bin/env python3
"""
AI Compliance Co-Pilot - app.py

A clean, modern Streamlit web application for navigating FSSAI regulations.
Provides an intelligent assistant interface for compliance questions.

Author: Python UI Developer
Date: July 2025
"""

import streamlit as st
import requests
import json

# Configure page settings
st.set_page_config(
    page_title="AI Compliance Co-Pilot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_BASE_URL = "http://127.0.0.1:8000"
SEARCH_ENDPOINT = f"{API_BASE_URL}/search"

def search_documents(query: str) -> dict:
    """
    Send search query to the backend API.
    
    Args:
        query: Search query string
        
    Returns:
        Dictionary containing the API response or error information
    """
    try:
        payload = {
            "query": query,
            "max_results": 5
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
            "error": "Could not connect to the backend server. Please ensure the API is running."
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

def main():
    """Main application function."""
    
    # Sidebar Content
    with st.sidebar:
        st.header("💡 How to Use")
        st.markdown("""
        1. In the main text area, type your compliance question about FSSAI regulations.
        2. Press 'Enter' to submit your query to the AI.
        3. Review the AI-generated answer that appears on the right.
        """)
        
        st.divider()
        
        st.header("❓ Example Queries")
        st.markdown("""
        * "What are the labeling requirements for packaged honey?"
        * "Tell me the licensing requirements for a new dairy processing unit."
        * "Summarize the regulations for food additives."
        * "What does the law say about importing food products?"
        """)
        
        st.caption("Disclaimer: This AI provides summaries and should not be considered legal advice. Always consult official FSSAI documents for final decisions.")
    
    # Main Area Content
    st.title("AI Compliance Co-Pilot 🚀")
    st.subheader("Your Intelligent Assistant for Navigating FSSAI Regulations")
    
    # Text input for user query
    col_input, col_button = st.columns([4, 1])
    
    with col_input:
        user_query = st.text_input(
            "Enter your compliance question here...",
            placeholder="What would you like to know about FSSAI regulations?",
            help="Type your question and press Enter or click Search to get AI-powered answers"
        )
    
    with col_button:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing to align with text input
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Core Logic - trigger on either button click or Enter key press
    if (user_query and user_query.strip()) or search_button:
        if not user_query or not user_query.strip():
            st.warning("⚠️ Please enter a question to search.")
            return
        # Create two columns for layout
        col1, col2 = st.columns([1, 1])
        
        # First column - AI Answer
        with col1:
            with st.spinner("Searching standards and generating answer..."):
                result = search_documents(user_query.strip())
            
            st.subheader("Answer:")
            
            if result["success"]:
                response_data = result["data"]
                st.markdown(response_data['answer'])
            else:
                st.error(f"Error: {result['error']}")
        
        # Second column - Source Documents
        with col2:
            st.subheader("Source Documents:")
            
            if result["success"]:
                response_data = result["data"]
                sources = response_data.get("sources", [])
                
                if sources:
                    st.markdown("**Referenced documents:**")
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**{i}.** {source}")
                else:
                    st.info("No source documents available for this query.")
            else:
                st.warning("Unable to retrieve source documents due to API error.")

if __name__ == "__main__":
    main()