"""
RAG Engine - Core Logic for Document Processing and Question Answering

This module handles:
1. Document loading and chunking
2. Embedding generation and vector storage
3. Semantic search and retrieval
4. Question answering with source attribution
"""

import os
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
# This reads OPENAI_API_KEY and other config values
load_dotenv()


class RAGEngine:
    """
    Retrieval-Augmented Generation Engine
    
    This class encapsulates all RAG functionality:
    - Document processing and chunking
    - Vector database management
    - Question answering with context
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the RAG engine
        
        Args:
            persist_directory: Where to store the vector database on disk
                             ChromaDB will save embeddings here so you don't
                             need to re-embed documents every time
        """
        # Storage location for vector database
        self.persist_directory = persist_directory
        
        # Initialize OpenAI embeddings model
        # This converts text into 1536-dimensional vectors
        # Cost: ~$0.0001 per 1000 tokens (very cheap)
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        )
        
        # Initialize the vector store (database for embeddings)
        # If it exists on disk, load it. Otherwise, starts empty.
        self.vectorstore = None
        
        # Initialize the LLM (Large Language Model) for answering questions
        # This is the "brain" that generates answers
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            temperature=0,  # 0 = deterministic (same input = same output)
                           # 1 = creative (more random, varied responses)
        )
        
        # Text splitter - breaks documents into digestible chunks
        # Why split? GPT has token limits, and smaller chunks = better retrieval
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,        # Each chunk = ~1000 characters
            chunk_overlap=200,      # Overlap ensures context isn't lost at boundaries
            length_function=len,    # How to measure chunk size
            separators=["\n\n", "\n", " ", ""]  # Try to split on paragraphs first,
                                                # then sentences, then words
        )
    
    def load_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Load and process PDF documents
        
        Args:
            file_paths: List of paths to PDF files
            
        Returns:
            List of document chunks with metadata
            
        Process:
        1. Load PDF and extract text
        2. Split into chunks
        3. Add metadata (source file, page numbers)
        """
        all_documents = []
        
        for file_path in file_paths:
            try:
                # PyPDFLoader extracts text from PDF
                # It preserves page numbers in metadata
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                
                # Split documents into chunks
                # Each chunk becomes a separate "document" in vector DB
                chunks = self.text_splitter.split_documents(documents)
                
                # Add to collection
                all_documents.extend(chunks)
                
                print(f"✅ Loaded {len(chunks)} chunks from {os.path.basename(file_path)}")
                
            except Exception as e:
                print(f"❌ Error loading {file_path}: {str(e)}")
                continue
        
        return all_documents
    
    def create_vector_store(self, documents: List[Any]) -> None:
        """
        Create vector database from documents
        
        Args:
            documents: List of document chunks to embed
            
        Process:
        1. Generate embeddings for each chunk (call OpenAI API)
        2. Store embeddings + original text in ChromaDB
        3. Persist to disk for future use
        
        Cost: ~$0.0001 per 1000 tokens
        For a 100-page PDF (~50,000 tokens), costs ~$0.005
        """
        if not documents:
            raise ValueError("No documents to process!")
        
        print(f"🔄 Creating embeddings for {len(documents)} chunks...")
        print("⏳ This may take a minute (calling OpenAI API)...")
        
        # Chroma.from_documents does several things:
        # 1. Calls OpenAI API to get embeddings for each chunk
        # 2. Stores embeddings in ChromaDB
        # 3. Links embeddings to original text and metadata
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        print("✅ Vector store created and persisted!")
    
    def load_existing_vectorstore(self) -> bool:
        """
        Load existing vector database from disk
        
        Returns:
            True if successful, False if no database exists
            
        Why useful? Avoids re-embedding documents (saves time and $)
        """
        try:
            # Try to load from disk
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            
            # Check if it has any data
            # _collection.count() returns number of vectors stored
            if self.vectorstore._collection.count() > 0:
                print(f"✅ Loaded existing vector store with {self.vectorstore._collection.count()} chunks")
                return True
            return False
            
        except Exception as e:
            print(f"No existing vector store found: {str(e)}")
            return False
    
    def query(self, question: str, k: int = 4) -> Dict[str, Any]:
        """
        Answer a question using RAG
        
        Args:
            question: User's question
            k: Number of document chunks to retrieve (default: 4)
               More chunks = more context but slower and more expensive
               
        Returns:
            Dictionary with 'answer' and 'sources'
            
        Process:
        1. Convert question to embedding
        2. Find k most similar document chunks (semantic search)
        3. Create prompt: "Given these documents [chunks], answer [question]"
        4. Send to GPT
        5. Return answer + sources
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized! Load or create documents first.")
        
        # Custom prompt template for better answers
        # This tells GPT HOW to answer questions
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Always cite which document/page the information came from.

        Context:
        {context}

        Question: {question}

        Helpful Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        # Create RetrievalQA chain
        # This orchestrates: retrieve → format prompt → call LLM
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # "stuff" = put all chunks in one prompt
                                # Other options: "map_reduce", "refine"
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": k}  # Retrieve top k chunks
            ),
            return_source_documents=True,  # Include sources in response
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        # Execute the query
        print(f"🔍 Searching for relevant chunks...")
        result = qa_chain({"query": question})
        
        # Format sources for display
        sources = []
        for doc in result["source_documents"]:
            sources.append({
                "content": doc.page_content[:200] + "...",  # First 200 chars
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown")
            })
        
        return {
            "answer": result["result"],
            "sources": sources
        }


# Example usage (for testing)
if __name__ == "__main__":
    # This block only runs if you execute: python rag_engine.py
    # Useful for testing the engine independently
    
    engine = RAGEngine()
    
    # Try to load existing database
    if not engine.load_existing_vectorstore():
        print("No existing database. Load documents first.")
    else:
        # Test query
        response = engine.query("What is this document about?")
        print(f"\nAnswer: {response['answer']}")
        print(f"\nSources: {len(response['sources'])} chunks used")




     

