from bioimageio_chatbot.utils import ChatbotExtension
from bioimageio_chatbot.knowledge_base import load_docs_store
from schema_agents import schema_tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from pydantic import Field, BaseModel
from typing import Any, Dict
import json
import requests
from markdownify import markdownify as md
import re
import os


class DocWithScore(BaseModel):
    """A document with an associated relevance score."""

    doc: str = Field(description="The document retrieved.")
    score: float = Field(description="The relevance score of the retrieved document.")
    metadata: Dict[str, Any] = Field(description="The metadata of the retrieved document.")

def load_eurobioimaging_base(db_path):
    docs_store_dict = {}
    print(f"Loading EuroBioImaging docs store from {db_path}")
    for collection in ['technologies', 'nodes']:
        try:
            docs_store = load_docs_store(db_path, f"eurobioimaging-{collection}")
            length = len(docs_store.docstore._dict.keys())
            assert length > 0, f"Please make sure the docs store {collection} is not empty."
            print(f"- Loaded {length} documents from {collection}")
            docs_store_dict[collection] = docs_store
        except Exception as e:
            print(f"Failed to load docs store for {collection}. Error: {e}")

    if len(docs_store_dict) == 0:
        raise Exception("No docs store is loaded, please make sure the docs store is not empty.")
    # load the node index
    with open(os.path.join(db_path, "eurobioimaging-node-index.json"), "r") as f:
        node_index = json.load(f)
    return docs_store_dict, node_index

def chunkify(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def create_eurobioimaging_vector_database(output_dir=None):
    if output_dir is None:
        output_dir = os.environ.get("BIOIMAGEIO_KNOWLEDGE_BASE_PATH", "./bioimageio-knowledge-base")
    os.makedirs(output_dir, exist_ok=True)
    
    EUROBIOIMAGING_TOKEN = os.getenv("EUROBIOIMAGING_TOKEN")
    assert EUROBIOIMAGING_TOKEN is not None, "Please set the EUROBIOIMAGING_TOKEN environment variable to access the EuroBioImaging API."
    
    embeddings = OpenAIEmbeddings()
    
    response = requests.get(f"https://www.eurobioimaging.eu/api-general-data.php?a={EUROBIOIMAGING_TOKEN}")
    collections = response.json()
    
    for name in ["technologies", "nodes"]:
        all_metadata = []
        all_contents = []
        for item in collections[name]:
            print(f"Fetch description from {item['url']}...")
            response = requests.get(f"https://www.eurobioimaging.eu/api-page-content.php?a={EUROBIOIMAGING_TOKEN}&url={item['url']}")
            content = md(response.text, heading_style="ATX")
            description = re.sub(r"\n{3,}", "\n\n", content).strip()
            
            # Generate embeddings for the item
            item_content = f"# {item['name']}\n\n{description}"
            all_contents.append(item_content)
            if "description" in item:
                item.pop("description")
            all_metadata.append(item)
        
        print(f"Embedding {len(all_contents)} documents...")
        all_embedding_pairs = []
        for chunk in chunkify(all_contents, 300):
            item_embeddings = embeddings.embed_documents(chunk)
            all_embedding_pairs.extend(zip(chunk, item_embeddings))

        # Create the FAISS index from all the embeddings
        vectordb = FAISS.from_embeddings(all_embedding_pairs, embeddings, metadatas=all_metadata)
        print("Saving the vector database...")
        vectordb.save_local(output_dir, index_name="eurobioimaging-" + name)
        print("Created a vector database from the downloaded documents.")

    node_index = {}
    for item in collections["nodes"]:
        node_index[item['node_id']] = item
    with open(os.path.join(output_dir, "eurobioimaging-node-index.json"), "w") as f:
        json.dump(node_index, f)
    

def create_tools(docs_store_dict, node_index):
    async def search_technology(
        keywords: str = Field(..., description="The keywords used for searching the technology in EuroBioImaging service index."),
        top_k: int = Field(..., description="Return the top_k number of search results")
    ):
        """Search Technology in EuroBioImaging service index"""
        results = []
        collection = "technologies"
        top_k = max(1, min(top_k, 15))
        docs_store = docs_store_dict[collection]

        print(f"Retrieving documents from database {collection} with keywords: {keywords}")
        results.append(
            await docs_store.asimilarity_search_with_relevance_scores(
                keywords, k=top_k
            )
        )
        docs_with_score = [
            DocWithScore(
                doc=doc.page_content,
                score=round(score, 2),
                metadata=doc.metadata, 
            )
            for results_with_scores in results
            for doc, score in results_with_scores
        ]
        # sort by relevance score
        docs_with_score = sorted(docs_with_score, key=lambda x: x.score, reverse=True)[
            : top_k
        ]

        if len(docs_with_score) > 2:
            print(
                f"Retrieved documents:\n{docs_with_score[0].doc[:20] + '...'} (score: {docs_with_score[0].score})\n{docs_with_score[1].doc[:20] + '...'} (score: {docs_with_score[1].score})\n{docs_with_score[2].doc[:20] + '...'} (score: {docs_with_score[2].score})"
            )
        else:
            print(f"Retrieved documents:\n{docs_with_score}")
        return docs_with_score


    async def get_node_details(
        node_id: str = Field(..., description="The EuroBioImaging node id"),
    ):
        """Get details of the EuroBioImaging node who provide services and technoliges to users"""
        if node_id in node_index:
            return node_index[node_id]
        else:
            return f"Node not found: {node_id}"
        

    async def search_node(
        keywords: str = Field(..., description="The keywords for searching the service nodes"),
        top_k: int = Field(..., description="Return the top_k number of search results")
    ):
        """Search a service node in the EuroBioImaging network"""
        results = []
        collection = "nodes"
        top_k = max(1, min(top_k, 15))
        docs_store = docs_store_dict[collection]

        print(f"Retrieving documents from database {collection} with query: {keywords}")
        results.append(
            await docs_store.asimilarity_search_with_relevance_scores(
                keywords, k=top_k
            )
        )

        docs_with_score = [
            DocWithScore(
                doc=doc.page_content,
                score=round(score, 2),
                metadata=doc.metadata,  
            )
            for results_with_scores in results
            for doc, score in results_with_scores
        ]
        # sort by relevance score
        docs_with_score = sorted(docs_with_score, key=lambda x: x.score, reverse=True)[
            : top_k
        ]

        if len(docs_with_score) > 2:
            print(
                f"Retrieved documents:\n{docs_with_score[0].doc[:20] + '...'} (score: {docs_with_score[0].score})\n{docs_with_score[1].doc[:20] + '...'} (score: {docs_with_score[1].score})\n{docs_with_score[2].doc[:20] + '...'} (score: {docs_with_score[2].score})"
            )
        else:
            print(f"Retrieved documents:\n{docs_with_score}")
        return docs_with_score
    return schema_tool(search_technology), schema_tool(search_node), schema_tool(get_node_details)


def get_extension():
    # collections = get_manifest()["collections"]
    knowledge_base_path = os.environ.get(
        "BIOIMAGEIO_KNOWLEDGE_BASE_PATH", "./bioimageio-knowledge-base"
    )
    assert (
        knowledge_base_path is not None
    ), "Please set the BIOIMAGEIO_KNOWLEDGE_BASE_PATH environment variable to the path of the knowledge base."
    
    # check if node_index exists
    if not os.path.exists(os.path.join(knowledge_base_path, "eurobioimaging-node-index.json")):
        EUROBIOIMAGING_TOKEN = os.getenv("EUROBIOIMAGING_TOKEN")
        if EUROBIOIMAGING_TOKEN is None:
            print("Warning: Disable EuroBioImaging extension because the EUROBIOIMAGING_TOKEN is not set.")
            return None
        print("Creating EuroBioImaging vector database...")
        create_eurobioimaging_vector_database(knowledge_base_path)
        
    docs_store_dict, node_index = load_eurobioimaging_base(knowledge_base_path)
    search_technology, search_node, get_node_details = create_tools(docs_store_dict, node_index)
    
    return ChatbotExtension(
        id="eurobioimaging",
        name="EuroBioImaging Service Index",
        description="Help users to find bioimaging services in the EuroBioimaging network; You can search by keywords for the imaging technology, then use the returned node_id to find out details about the service providing node in the EuroBioimang network",
        tools=dict(
            search_technology=search_technology,
            search_node=search_node,
            get_node_details=get_node_details
        )
    )

if __name__ == "__main__":
    import asyncio
    async def main():
        extension = get_extension()
        query = "mouse embryos"
        top_k = 2
        print(await extension.tools["search_technology"](keywords=query, top_k=top_k))
    asyncio.run(main())