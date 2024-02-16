import httpx
from bs4 import BeautifulSoup
from langchain.schema import Document

from .langchain_websearch import LangchainCompressor


async def search_duckduckgo(query: str, langchain_compressor: LangchainCompressor,
                                max_results: int, similarity_threshold: float, instant_answers: bool,
                                chunk_size: int, num_results_to_process: int):
    
    from duckduckgo_search import AsyncDDGS
    documents = []
    query = query.strip("\"'")
    async with AsyncDDGS() as ddgs:
        if instant_answers:
            answer_list = []
            async for answer in ddgs.answers(query):
                answer_list.append(answer)
            if answer_list:
                max_results -= 1  # We already have 1 result now
                answer_dict = answer_list[0]
                instant_answer_doc = Document(page_content=answer_dict["text"],
                                              metadata={"source": answer_dict["url"]})
                documents.append(instant_answer_doc)

        results = []
        result_urls = []
        async for result in ddgs.text(query, region='wt-wt', safesearch='moderate', timelimit=None,
                                max_results=num_results_to_process):
            results.append(result)
            result_urls.append(result["href"])

    documents.extend(await langchain_compressor.faiss_embedding_query_urls(query, result_urls,
                                                               num_results=num_results_to_process,
                                                               similarity_threshold=similarity_threshold,
                                                               chunk_size=chunk_size))
    
    if not documents:    # Fall back to old simple search rather than returning nothing
        print("LLM_Web_search | Could not find any page content "
              "similar enough to be extracted, using basic search fallback...")
        data = results[:max_results]
        docs = []
        for d in data:
            docs.append({"title": d['title'], "body": d['body'], "href": d['href']})
        return docs
    
    docs = []
    for doc in documents[:max_results]:
        docs.append({"content": doc.page_content, "url": doc.metadata["source"]})
    return docs


async def get_webpage_content(url: str) -> str:
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
