"""
RAG (Retrieval Augmented Generation) functionality for document processing
"""
import streamlit as st
import PyPDF2
import tempfile
import os
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid

class DocumentProcessor:
    """Process and store documents for RAG"""
    
    def __init__(self):
        """Initialize document processor with embedding model"""
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize ChromaDB
            self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
            
            # Create or get collection
            try:
                self.collection = self.chroma_client.get_collection("documents")
            except:
                self.collection = self.chroma_client.create_collection("documents")
                
        except Exception as e:
            st.error(f"Error initializing document processor: {str(e)}")
            self.embedding_model = None
            self.collection = None
    
    def extract_text_from_pdf(self, uploaded_file) -> str:
        """
        Extract text from uploaded PDF file
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Extracted text content
        """
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Extract text from PDF
            text = ""
            with open(tmp_file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            return text.strip()
            
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to chunk
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to end at a sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                
                if last_period > chunk_size * 0.7:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1
                elif last_newline > chunk_size * 0.7:
                    chunk = chunk[:last_newline]
                    end = start + last_newline
            
            chunks.append(chunk.strip())
            start = max(start + chunk_size - overlap, end)
            
            if start >= len(text):
                break
        
        return [chunk for chunk in chunks if chunk]
    
    def store_document(self, filename: str, text: str) -> bool:
        """
        Store document chunks in vector database
        
        Args:
            filename: Name of the document
            text: Document text content
            
        Returns:
            True if successful, False otherwise
        """
        if not self.embedding_model or not self.collection:
            return False
            
        try:
            # Clear existing documents with same filename
            try:
                existing_docs = self.collection.get(where={"filename": filename})
                if existing_docs['ids']:
                    self.collection.delete(ids=existing_docs['ids'])
            except:
                pass  # Collection might be empty
            
            # Chunk the text
            chunks = self.chunk_text(text)
            
            if not chunks:
                return False
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(chunks).tolist()
            
            # Prepare metadata
            ids = [str(uuid.uuid4()) for _ in chunks]
            metadatas = [{"filename": filename, "chunk_index": i} for i in range(len(chunks))]
            
            # Store in ChromaDB
            self.collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            return True
            
        except Exception as e:
            st.error(f"Error storing document: {str(e)}")
            return False
    
    def search_documents(self, query: str, n_results: int = 3) -> List[Dict]:
        """
        Search for relevant document chunks
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of relevant document chunks with metadata
        """
        if not self.embedding_model or not self.collection:
            return []
            
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Debug: Check what we got
            print(f"Debug - Search query: {query}")
            print(f"Debug - Results found: {len(results['documents'][0]) if results['documents'] and results['documents'][0] else 0}")
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                        'distance': results['distances'][0][i] if results['distances'] and results['distances'][0] else 0
                    })
                    print(f"Debug - Found chunk {i}: {doc[:100]}...")
            
            return formatted_results
            
        except Exception as e:
            st.error(f"Error searching documents: {str(e)}")
            return []
    
    def get_stored_documents(self) -> List[str]:
        """
        Get list of stored document filenames
        
        Returns:
            List of document filenames
        """
        if not self.collection:
            return []
            
        try:
            all_docs = self.collection.get()
            if all_docs['metadatas']:
                filenames = set()
                for metadata in all_docs['metadatas']:
                    if 'filename' in metadata:
                        filenames.add(metadata['filename'])
                return list(filenames)
            return []
            
        except Exception as e:
            st.error(f"Error getting stored documents: {str(e)}")
            return []
    
    # Compatibility methods for the chat interface
    def process_and_store_document(self, uploaded_file, document_name: str = None) -> bool:
        """Process uploaded document and store in vector database."""
        if uploaded_file is None:
            return False
        
        # Extract text from document
        text = self.extract_text_from_pdf(uploaded_file)
        if not text:
            return False
        
        # Store document
        doc_name = document_name or uploaded_file.name
        success = self.store_document(doc_name, text)
        
        return success
    
    def retrieve_relevant_context(self, query: str, n_results: int = 3):
        """Retrieve relevant context for a query."""
        results = self.search_documents(query, n_results)
        
        if not results:
            return "", []
        
        # Combine relevant texts
        context_texts = [result['content'] for result in results]
        combined_context = "\n\n".join(context_texts)
        
        # Format results for compatibility
        formatted_results = []
        for result in results:
            formatted_results.append({
                'text': result['content'],
                'distance': result.get('distance', 0),
                'metadata': result.get('metadata', {})
            })
        
        return combined_context, formatted_results
    
    def get_system_prompt_with_context(self, context: str) -> str:
        """Create system prompt with retrieved context."""
        if not context.strip():
            return "You are a helpful AI assistant. Please answer the user's questions to the best of your ability."
        
        return f"""You are a helpful AI assistant with access to document content. The user has uploaded documents and you have access to relevant sections.

IMPORTANT: You have the following document content available to answer the user's question:

--- DOCUMENT CONTENT ---
{context}
--- END DOCUMENT CONTENT ---

Use this document content to answer the user's questions. The user is asking about content from their uploaded document. You should reference and use the information provided above. Do not ask for the document or say you don't have access to it - you have the relevant content above."""
    
    def clear_documents(self):
        """Clear all stored documents."""
        if self.collection:
            try:
                # Get all items and delete them
                all_items = self.collection.get()
                if all_items['ids']:
                    self.collection.delete(ids=all_items['ids'])
                st.success("Document collection cleared!")
            except Exception as e:
                st.error(f"Error clearing collection: {str(e)}")
    
    def get_collection_info(self) -> Dict[str, any]:
        """Get information about current collection."""
        if not self.collection:
            return {"count": 0, "documents": []}
        
        try:
            all_items = self.collection.get()
            documents = self.get_stored_documents()
            
            # Debug: Print what we have
            print(f"Debug - Collection has {len(all_items['ids']) if all_items['ids'] else 0} items")
            if all_items['documents']:
                print(f"Debug - First document chunk: {all_items['documents'][0][:100]}...")
            
            return {
                "count": len(all_items['ids']) if all_items['ids'] else 0,
                "documents": documents
            }
        except Exception as e:
            st.error(f"Error getting collection info: {str(e)}")
            return {"count": 0, "documents": []}


# Initialize RAG system
@st.cache_resource
def get_rag_system():
    """Get cached RAG system instance."""
    return DocumentProcessor()