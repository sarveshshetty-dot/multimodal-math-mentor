import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

def get_vector_store(persist_directory="data/faiss_index"):
    """
    Initializes and loads the FAISS vector store using local Sentence-Transformers.
    """
    model_name = "all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    if os.path.exists(os.path.join(persist_directory, "index.faiss")):
        return FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True)
    return None
