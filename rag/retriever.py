import os
import streamlit as st
from .vector_store import get_vector_store

def retrieve_context(query: str, k: int = 3, vectorstore=None) -> str:
    """
    Retrieves the top-k most relevant contexts from the RAG knowledge base.
    """
    try:
        # Strict type checking for query
        if not isinstance(query, str):
            query = str(query) if query is not None else ""
        
        if not query.strip():
            return ""

        if vectorstore is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            persist_dir = os.path.join(base_dir, "data/faiss_index")
            try:
                vectorstore = get_vector_store(persist_dir)
            except Exception as e:
                st.error(f"Error loading vector store: {e}")
                return ""
        
        # Check if vectorstore actually supports as_retriever
        if vectorstore is None or not hasattr(vectorstore, "as_retriever"):
            return ""
            
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        
        # Use invoke if available, fallback to get_relevant_documents for older LangChain
        if hasattr(retriever, "invoke"):
            docs = retriever.invoke(query)
        elif hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(query)
        else:
            return ""
        
        if not docs:
            return ""
            
        # Join the retrieved chunks into a single readable string
        # Ensure page_content is a string
        formatted_context = "\n\n---\n\n".join([str(doc.page_content) for doc in docs if hasattr(doc, 'page_content')])
        return formatted_context
    except Exception as e:
        st.error(f"RAG Retrieval Error: {e}")
        return ""
