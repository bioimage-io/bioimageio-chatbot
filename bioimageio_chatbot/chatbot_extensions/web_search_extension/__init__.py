
from schema_agents import schema_tool
from bioimageio_chatbot.utils import ChatbotExtension
from pydantic import Field
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from .langchain_websearch import LangchainCompressor

default_langchain_compressor = None

@schema_tool
async def search_web(query: str=Field(description="space separated keywords for the duckduckgo search engine"), max_results: int = Field(description="maximum number of results to return")):
    """Search the web for information using duckduckgo."""
    from duckduckgo_search import AsyncDDGS
    query = query.strip("\"'")
    results = await AsyncDDGS(proxy=None).atext(query, region='wt-wt', safesearch='moderate', timelimit=None,
                            max_results=max_results)
    if not results:
        return "No relevant information found."
    docs = []
    for d in results:
        docs.append({"title": d['title'], "body": d['body'], "url": d['href']})
    return docs

@schema_tool
async def browse_web_pages(query: str=Field(description="keywords or a sentence describing the information to be retrieved"), urls: list[str]=Field(description="list of web page urls to analyse"), num_results_to_process: Optional[int]=Field(5, description="number of results to process")):
    """Read web pages and return compressed documents with most relevant information."""
    global default_langchain_compressor
    default_langchain_compressor = default_langchain_compressor or LangchainCompressor(device="cpu")

    documents = await default_langchain_compressor.faiss_embedding_query_urls(query, urls,
                                                               num_results=num_results_to_process)
    
    if not documents:    # Fall back to old simple search rather than returning nothing
        print("LLM_Web_search | Could not find any page content "
              "similar enough to be extracted, using basic search fallback...")
        return "No relevant information found."
    #return the json serializable documents
    return [doc.page_content + '\nsource: ' + doc.metadata.get('source') for doc in documents]

@schema_tool
async def read_webpage(url: str=Field(description="the web url to read")) -> str:
    """Read the full content of a web page converted to plain text."""
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)

    soup = BeautifulSoup(response.content, features="lxml")
    for script in soup(["script", "style"]):
        script.extract()

    strings = soup.stripped_strings
    return '\n'.join([s.strip() for s in strings])


def get_extension():
    return ChatbotExtension(
        id="web",
        name="Search Web",
        description="Search the web for information using duckduckgo. Search by keywords and returns a list of relevant documents.",
        tools=dict(search=search_web, browse=browse_web_pages)
    )
