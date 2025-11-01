"""
Smart Document RAG Assistant - Streamlit UI

This is the user interface for interacting with the RAG engine.
Users can upload PDFs, ask questions, and see answers with sources.
"""

import streamlit as st
import os
from pathlib import Path
from rag_engine import RAGEngine

# Streamlit page configuration
# This MUST be the first Streamlit command
st.set_page_config(
    page_title="Smart Document Assistant",  # Browser tab title
    page_icon="📚",                         # Browser tab icon
    layout="wide",                          # Use full width
    initial_sidebar_state="expanded"        # Show sidebar by default
)

# Custom CSS for better styling
# Streamlit's default styling is basic - this makes it prettier
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """
    Initialize Streamlit session state variables
    
    Session state persists data across reruns (when user interacts with UI)
    Without this, data would reset every time user clicks a button
    
    Think of it like React's useState or Vue's reactive data
    """
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = RAGEngine()
    
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


def save_uploaded_file(uploaded_file) -> str:
    """
    Save uploaded file to disk temporarily
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Path to saved file
        
    Why? PyPDFLoader needs a file path, not a file object
    """
    # Create upload directory if it doesn't exist
    upload_dir = Path("uploaded_docs")
    upload_dir.mkdir(exist_ok=True)
    
    # Save file
    file_path = upload_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return str(file_path)


def main():
    """
    Main application logic
    
    Streamlit runs this function top-to-bottom every time:
    - User uploads a file
    - User clicks a button
    - User types in an input
    
    This is why we need session_state to persist data
    """
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("Smart Document RAG Assistant")
    st.markdown("Upload your documents and ask questions. Powered by GPT and semantic search.")
    
    # Sidebar for document management
    with st.sidebar:
        st.header("Document Management")
        
        # File uploader
        # accept_multiple_files=True allows selecting multiple PDFs at once
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Select one or more PDF files to analyze"
        )
        
        # Process uploaded files
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                with st.spinner("Processing documents... This may take a minute."):
                    try:
                        # Save uploaded files to disk
                        file_paths = []
                        for uploaded_file in uploaded_files:
                            file_path = save_uploaded_file(uploaded_file)
                            file_paths.append(file_path)
                            st.success(f"Saved: {uploaded_file.name}")
                        
                        # Load and process documents
                        documents = st.session_state.rag_engine.load_documents(file_paths)
                        
                        if documents:
                            # Create vector store (generate embeddings)
                            st.session_state.rag_engine.create_vector_store(documents)
                            st.session_state.documents_loaded = True
                            
                            st.success(f" Successfully processed {len(documents)} document chunks!")
                            st.balloons()  # Fun animation!
                        else:
                            st.error("No documents could be processed. Check file format.")
                    
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
        
        # Show status
        st.divider()
        if st.session_state.documents_loaded:
            st.success("Documents loaded and ready!")
            
            # Option to clear database
            if st.button("Clear Database"):
                # This doesn't actually delete from disk - just resets session
                # To fully clear, you'd need to delete chroma_db/ folder
                st.session_state.rag_engine = RAGEngine()
                st.session_state.documents_loaded = False
                st.session_state.chat_history = []
                st.info("Database cleared. Upload new documents to start fresh.")
                st.rerun()  # Refresh the page
        else:
            st.info("Upload documents to get started")
        
        # Display API info
        st.divider()
        st.caption("Using OpenAI API")
        st.caption(f"Model: {os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')}")
    
    # Main content area - Q&A Interface
    if st.session_state.documents_loaded:
        st.header("Ask Questions")
        
        # Question input
        # form prevents page reload on every keystroke
        with st.form(key="question_form", clear_on_submit=True):
            question = st.text_input(
                "What would you like to know?",
                placeholder="e.g., What are the main findings of this research?",
                help="Ask anything about your uploaded documents"
            )
            
            # Advanced options (collapsible)
            with st.expander("Advanced Options"):
                k = st.slider(
                    "Number of chunks to retrieve",
                    min_value=1,
                    max_value=10,
                    value=4,
                    help="More chunks = more context but slower and more expensive"
                )
            
            # Submit button
            submitted = st.form_submit_button("🔍 Ask Question", type="primary")
        
        # Process question
        if submitted and question:
            with st.spinner("Thinking..."):
                try:
                    # Query the RAG engine
                    response = st.session_state.rag_engine.query(question, k=k)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": response["answer"],
                        "sources": response["sources"]
                    })
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Display chat history (most recent first)
        if st.session_state.chat_history:
            st.divider()
            st.header("Conversation History")
            
            # Reverse to show newest first
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.container():
                    # Question
                    st.markdown(f"** Question {len(st.session_state.chat_history) - i}:**")
                    st.info(chat["question"])
                    
                    # Answer
                    st.markdown("** Answer:**")
                    st.success(chat["answer"])
                    
                    # Sources
                    if chat["sources"]:
                        st.markdown("** Sources:**")
                        for j, source in enumerate(chat["sources"], 1):
                            with st.expander(f"Source {j}: {Path(source['source']).name} (Page {source['page']})"):
                                st.markdown(f"```\n{source['content']}\n```")
                    
                    st.divider()
        
    else:
        # Welcome screen when no documents loaded
        st.info("Upload documents in the sidebar to get started!")
        
        # Instructions
        st.markdown("""
        ### How to use:
        1. **Upload PDFs** in the sidebar (one or multiple files)
        2. **Click "Process Documents"** to analyze them
        3. **Ask questions** about your documents
        4. **View answers** with source citations
        
        ### What is RAG?
        Retrieval-Augmented Generation (RAG) combines:
        - **Retrieval**: Finding relevant information from your documents
        - **Generation**: Using AI to answer questions based on that information
        
        This means GPT can answer questions about YOUR documents, not just its training data!
        
        ### Features:
        -  Semantic search (understands meaning, not just keywords)
        -  Source citations (see where answers come from)
        -  Multiple document support
        -  Conversation history
        -  Adjustable retrieval settings
        """)
        
        # Example questions
        with st.expander(" Example Questions to Try"):
            st.markdown("""
            - What are the main conclusions of this document?
            - Summarize the key findings in bullet points
            - What methodology was used in this research?
            - Are there any limitations mentioned?
            - What recommendations are provided?
            - Compare and contrast [topic A] and [topic B]
            """)


# Entry point
# This is what runs when you execute: streamlit run app.py
if __name__ == "__main__":
    main()