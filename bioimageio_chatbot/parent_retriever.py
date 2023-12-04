from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.storage import InMemoryStore
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.vectorstores.faiss import DistanceStrategy
from langchain.docstore.in_memory import InMemoryDocstore
import faiss
import uuid
import pickle
from typing import List

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.pydantic_v1 import Field
from langchain.schema import BaseRetriever, BaseStore, Document
from langchain.schema.vectorstore import VectorStore


class MultiVectorRetriever(BaseRetriever):
    """Retrieve from a set of multiple embeddings for the same document."""

    vectorstore: VectorStore
    """The underlying vectorstore to use to store small chunks
    and their embedding vectors"""
    docstore: BaseStore[str, Document]
    """The storage layer for the parent documents"""
    id_key: str = "doc_id"
    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass to the search function."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """
        sub_docs = self.vectorstore.similarity_search(query, k=8, **self.search_kwargs)
        # We do this to maintain the order of the ids that are returned
        ids = []
        for d in sub_docs:
            if d.metadata[self.id_key] not in ids:
                ids.append(d.metadata[self.id_key])
        docs = self.docstore.mget(ids)
        return [d for d in docs if d is not None]
    
    
    
def build_parent_retriever(docs, index_name, embeddings, child_splitter, save_dir=None, id_key = "doc_id"):
    index = faiss.IndexFlatL2(1536)
    vectorstore = FAISS(
        embeddings,
        index,
        InMemoryDocstore(),
        {},
        normalize_L2=False,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )
    # The storage layer for the parent documents
    store = InMemoryStore()
    
    # The retriever (empty to start)
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    doc_ids = [str(uuid.uuid4()) for _ in docs]
    sub_docs = []
    for i, doc in enumerate(docs):
        _id = doc_ids[i]
        _sub_docs = child_splitter.split_documents([doc])
        for _doc in _sub_docs:
            _doc.metadata[id_key] = _id
        sub_docs.extend(_sub_docs)
    retriever.vectorstore.add_documents(sub_docs)
    retriever.docstore.mset(list(zip(doc_ids, docs)))
    if save_dir is not None:
        # save the vectorstore
        vectorstore.save_local(save_dir, index_name=index_name)
        # save the store using pickle
        with open(f"{save_dir}/{index_name}_docstore.pkl", "wb") as f:
            pickle.dump(store, f)
    return retriever
    

def load_parent_retriever(output_dir, index_name, embeddings, id_key = "doc_id"):
    vectorstore = FAISS.load_local(index_name=index_name, folder_path=output_dir, embeddings=embeddings)
    with open(f"{output_dir}/{index_name}_docstore.pkl", "rb") as f:
        docstore = pickle.load(f)
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=id_key,
    )
    return retriever

if __name__ == "__main__":
    loaders = [
        TextLoader("/home/alalulu/workspace/chatbot_bmz/chatbot/data/main.zip-unzipped/bioimage.io-main/docs/faqs/README.md"),
        TextLoader("/home/alalulu/workspace/chatbot_bmz/chatbot/data/main.zip-unzipped/bioimage.io-main/docs/README.md"),
    ]

    docs = []
    for l in loaders:
        docs.extend(l.load())
        
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    docs = parent_splitter.split_documents(docs)
    child_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    
    retriever = build_parent_retriever(docs, index_name="test", embeddings=OpenAIEmbeddings(), child_splitter=child_splitter, save_dir="./data/test_parent_retriever")
    sub_docs = retriever.vectorstore.similarity_search("Community Partner")
    print(sub_docs[0].page_content)
    retrieved_docs = retriever.get_relevant_documents("Community Partner")
    print(retrieved_docs[0].page_content)
    
    retriever = load_parent_retriever("./data/test_parent_retriever", index_name="test", embeddings=OpenAIEmbeddings())
    sub_docs = retriever.vectorstore.similarity_search("Community Partner")
    print(sub_docs[0].page_content)
    retrieved_docs = retriever.get_relevant_documents("Community Partner")
    print(retrieved_docs[0].page_content)
