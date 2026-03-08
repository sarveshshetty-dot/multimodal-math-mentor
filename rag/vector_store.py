import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

def get_embeddings():
    """
    Initializes the embedding model once.
    """
    model_name = "all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    return HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def get_vector_store(persist_directory="data/faiss_index", embeddings=None):
    """
    Loads the FAISS vector store. If embeddings are not provided, it initializes them.
    """
    import streamlit as st
    try:
        if embeddings is None:
            embeddings = get_embeddings()
        
        index_path = os.path.join(persist_directory, "index.faiss")
        if os.path.exists(index_path):
            return FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True)
        else:
            st.warning(f"FAISS index not found at {index_path}")
            return None
    except Exception as e:
        st.error(f"Error in get_vector_store: {e}")
        return None
