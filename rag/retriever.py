import os
from .vector_store import get_vector_store

def retrieve_context(query: str, k: int = 3, vectorstore=None) -> str:
    """
    Retrieves the top-k most relevant contexts from the RAG knowledge base.
    """
    if vectorstore is None:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        persist_dir = os.path.join(base_dir, "data/faiss_index")
        vectorstore = get_vector_store(persist_dir)
    
    if not vectorstore:
        return ""
        
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    
    if not docs:
        return ""
        
    # Join the retrieved chunks into a single readable string
    formatted_context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    return formatted_context
