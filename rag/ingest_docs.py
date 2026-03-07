import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_knowledge_base(kb_dir="data/knowledge_base", persist_directory="data/faiss_index"):
    """
    Chunks markdown files from the knowledge base and embeds them into a FAISS index.
    """
    os.makedirs(persist_directory, exist_ok=True)
    
    docs = []
    base_dir = os.path.dirname(os.path.dirname(__file__))
    full_kb_dir = os.path.join(base_dir, kb_dir)
    
    md_files = glob.glob(os.path.join(full_kb_dir, "*.md"))
    if not md_files:
        print(f"No documents found in {full_kb_dir}.")
        return False
        
    for file_path in md_files:
        loader = TextLoader(file_path, encoding='utf-8')
        docs.extend(loader.load())
        
    # Chunking using Markdown-aware splitter
    splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)
    
    print(f"Loaded {len(docs)} documents. Split into {len(splits)} chunks.")
    
    # Initialize local embeddings and create FAISS vector store
    model_name = "all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # Save standard index locally
    full_persist_dir = os.path.join(base_dir, persist_directory)
    vectorstore.save_local(full_persist_dir)
    print(f"Saved FAISS index to {full_persist_dir}.")
    
    return True

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    ingest_knowledge_base()
