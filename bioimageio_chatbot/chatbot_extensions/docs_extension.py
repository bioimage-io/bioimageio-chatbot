import os
import asyncio
from functools import partial
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
from bioimageio_chatbot.knowledge_base import load_knowledge_base
from bioimageio_chatbot.utils import get_manifest
from bioimageio_chatbot.utils import ChatbotExtension


class DocWithScore(BaseModel):
    """A document with an associated relevance score."""

    doc: str = Field(description="The document retrieved.")
    score: float = Field(description="The relevance score of the retrieved document.")


async def get_schema(collection):
    class DocumentRetrievalInput(BaseModel):
        """Searching knowledge base for relevant documents."""
        query: str = Field(
            description="The query used to retrieve documents related to the user's request. It should be a sentence which will be used to match descriptions using the OpenAI text embedding to match document chunks in a vector database."
        )
        top_k: int = Field(
            3,
            description="The maximum number of search results to return. Should use a small number to avoid overwhelming the user.",
        )

    channel_id = collection["id"]
    base_url = collection.get("base_url")
    if base_url:
        base_url_prompt = f" The documentation is available at {base_url}."
    else:
        base_url_prompt = ""
    DocumentRetrievalInput.__name__ = "Search" + title_case(channel_id)
    DocumentRetrievalInput.__doc__ = f"""Searching documentation for {channel_id}: {collection['description']}.{base_url_prompt}"""
    return DocumentRetrievalInput.schema()

async def run_extension(docs_store_dict, channel_id, req):
    channel_results = []
    # channel_urls = []
    # limit top_k from 1 to 15
    req.top_k = max(1, min(req.top_k, 15))
    docs_store = docs_store_dict[channel_id]

    print(f"Retrieving documents from database {channel_id} with query: {req.query}")
    channel_results.append(await docs_store.asimilarity_search_with_relevance_scores(
        req.query, k=req.top_k
    ))

        
    docs_with_score = [
        DocWithScore(
            doc=doc.page_content, score=round(score, 2), metadata=doc.metadata #, base_url=base_url
        )
        for results_with_scores in channel_results
        for doc, score in results_with_scores
    ]
    # sort by relevance score
    docs_with_score = sorted(docs_with_score, key=lambda x: x.score, reverse=True)[:req.top_k]
    
    if len(docs_with_score) > 2:
        print(
            f"Retrieved documents:\n{docs_with_score[0].doc[:20] + '...'} (score: {docs_with_score[0].score})\n{docs_with_score[1].doc[:20] + '...'} (score: {docs_with_score[1].score})\n{docs_with_score[2].doc[:20] + '...'} (score: {docs_with_score[2].score})"
        )
    else:
        print(f"Retrieved documents:\n{docs_with_score}")
    return docs_with_score


def title_case(s):
    return s.replace(".", " ").replace("-", " ").title().replace(" ", "")
    
def get_extensions():
    collections = get_manifest()["collections"]
    knowledge_base_path = os.environ.get("BIOIMAGEIO_KNOWLEDGE_BASE_PATH", "./bioimageio-knowledge-base")
    assert knowledge_base_path is not None, "Please set the BIOIMAGEIO_KNOWLEDGE_BASE_PATH environment variable to the path of the knowledge base."
    if not os.path.exists(knowledge_base_path):
        print(f"The knowledge base is not found at {knowledge_base_path}, will download it automatically.")
        os.makedirs(knowledge_base_path, exist_ok=True)
    
    knowledge_base_path = os.environ.get(
        "BIOIMAGEIO_KNOWLEDGE_BASE_PATH", "./bioimageio-knowledge-base"
    )
    docs_store_dict = load_knowledge_base(knowledge_base_path)
    return [
        ChatbotExtension(
            name="SearchDocs"+title_case(collection["id"]),
            description="Documentation for "+collection["id"] + "\n" + collection["description"],
            get_schema=partial(get_schema, collection),
            execute=partial(run_extension, docs_store_dict, collection["id"]),
        )
        for collection in collections
    ]
