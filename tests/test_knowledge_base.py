from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def test_knowledge_base():
    """Test the knowledge base"""
    vectordb = FAISS.load_local(folder_path="./bioimageio-knowledge-base", index_name="bioimage.io", embeddings=OpenAIEmbeddings())
    retriever = vectordb.as_retriever(score_threshold=0.4)
    items = retriever.get_relevant_documents("community partner", verbose=True)
    assert len(items) > 0