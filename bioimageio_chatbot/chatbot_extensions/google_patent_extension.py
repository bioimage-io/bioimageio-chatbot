
from serpapi import GoogleSearch
import os
import httpx
import asyncio
from typing import Dict, Tuple, Optional
from pydantic import BaseModel, Field
from bioimageio_chatbot.utils import ChatbotExtension

SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")

async def download_json(url: str, params: None) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()

class GooglePatentSearchParameters(BaseModel):
    """Parameters for searching Google Patents"""
    query: str = Field(..., description="The search query string.")
    num: Optional[int] = Field(10, ge=10, le=100, description="The maximum number of results to return, must be between 10 and 100 only")
    start: Optional[int] = Field(0, gt=0, description="Search result offset")

class GooglePatentReadParameters(BaseModel):
    """Parameters for reading detailed information about a patent from Google Patents"""
    patent_id: str = Field(..., description="The patent id to read")
    max_claim_count: Optional[int] = Field(10, description="The maximum number of claim to return")
    
async def search_google_patent(req: GooglePatentSearchParameters) -> list:
    """Query through SerpAPI and return the results async."""
    query = req.query
    def construct_url_and_params() -> Tuple[str, Dict[str, str]]:
        params = {
            "engine": "google_patents",
            "google_domain": "google.com",
            "gl": "us",
            "hl": "en",
            "api_key": SERPAPI_API_KEY,
            "q": query,
            "num": req.num,
            "start": req.start,
        }
        params["source"] = "python"
        params["serp_api_key"] = SERPAPI_API_KEY
        params["output"] = "json"
        url = "https://serpapi.com/search"
        return url, params

    url, params = construct_url_and_params()
    res = await download_json(url, params)
    processed_snippets = await process_response(res)
    return processed_snippets

async def read_a_google_patent(req: GooglePatentReadParameters) -> dict:
    """"Read detailed information about a patent from Google Patents."""
    if not req.patent_id.startswith("patent/"):
        req.patent_id = f"patent/{req.patent_id}/en"
        
    url = f"https://serpapi.com/search.json?engine=google_patents_details&patent_id={req.patent_id}"
    res = await download_json(url, params={"api_key":SERPAPI_API_KEY})
    if "error" in res.keys():
        raise ValueError(f"Got error from SerpAPI: {res['error']}")
    abstract = res.get("abstract")
    claims = res["claims"][:req.max_claim_count]
    return {"abstract": abstract, "claims": claims}

         
async def process_response(res: dict) -> list:
    """Process response from SerpAPI."""
    if "error" in res.keys():
        raise ValueError(f"Got error from SerpAPI: {res['error']}")
    if "answer_box_list" in res.keys():
        res["answer_box"] = res["answer_box_list"]
    if "answer_box" in res.keys():
        answer_box = res["answer_box"]
        if isinstance(answer_box, list):
            answer_box = answer_box[0]
        if "result" in answer_box.keys():
            return answer_box["result"]
        elif "answer" in answer_box.keys():
            return answer_box["answer"]
        elif "snippet" in answer_box.keys():
            return answer_box["snippet"]
        elif "snippet_highlighted_words" in answer_box.keys():
            return answer_box["snippet_highlighted_words"]
        else:
            answer = {}
            for key, value in answer_box.items():
                if not isinstance(value, (list, dict)) and not (
                    isinstance(value, str) and value.startswith("http")
                ):
                    answer[key] = value
            return str(answer)
    elif "events_results" in res.keys():
        return res["events_results"][:10]
    elif "sports_results" in res.keys():
        return res["sports_results"]
    elif "top_stories" in res.keys():
        return res["top_stories"]
    elif "news_results" in res.keys():
        return res["news_results"]
    elif "jobs_results" in res.keys() and "jobs" in res["jobs_results"].keys():
        return res["jobs_results"]["jobs"]
    elif (
        "shopping_results" in res.keys()
        and "title" in res["shopping_results"][0].keys()
    ):
        return res["shopping_results"][:3]
    elif "questions_and_answers" in res.keys():
        return res["questions_and_answers"]
    elif (
        "popular_destinations" in res.keys()
        and "destinations" in res["popular_destinations"].keys()
    ):
        return res["popular_destinations"]["destinations"]
    elif "top_sights" in res.keys() and "sights" in res["top_sights"].keys():
        return res["top_sights"]["sights"]
    elif (
        "images_results" in res.keys()
        and "thumbnail" in res["images_results"][0].keys()
    ):
        return str([item["thumbnail"] for item in res["images_results"][:10]])

    snippets = []
    if "knowledge_graph" in res.keys():
        knowledge_graph = res["knowledge_graph"]
        title = knowledge_graph["title"] if "title" in knowledge_graph else ""
        if "description" in knowledge_graph.keys():
            snippets.append(knowledge_graph["description"])
        for key, value in knowledge_graph.items():
            if (
                isinstance(key, str)
                and isinstance(value, str)
                and key not in ["title", "description"]
                and not key.endswith("_stick")
                and not key.endswith("_link")
                and not value.startswith("http")
            ):
                snippets.append(f"{title} {key}: {value}.")

    for organic_result in res.get("organic_results", []):
        snippets.append({
            "patent_id" : organic_result["patent_id"],
            "title" : organic_result["title"],
            # "url" : organic_result["serpapi_link"],
            "snippet" : organic_result["snippet"],
            # "assignee" : organic_result["assignee"],
            # "publication_number" : organic_result["publication_number"],
            "pdf" : organic_result["pdf"],
        })
        # if "snippet" in organic_result.keys():
        #     snippets.append(organic_result["snippet"])
        # elif "snippet_highlighted_words" in organic_result.keys():
        #     snippets.append(organic_result["snippet_highlighted_words"])
        # elif "rich_snippet" in organic_result.keys():
        #     snippets.append(organic_result["rich_snippet"])
        # elif "rich_snippet_table" in organic_result.keys():
        #     snippets.append(organic_result["rich_snippet_table"])
        # elif "link" in organic_result.keys():
        #     snippets.append(organic_result["link"])

    if "buying_guide" in res.keys():
        snippets.append(res["buying_guide"])
    if "local_results" in res and isinstance(res["local_results"], list):
        snippets += res["local_results"]
    if (
        "local_results" in res.keys()
        and isinstance(res["local_results"], dict)
        and "places" in res["local_results"].keys()
    ):
        snippets.append(res["local_results"]["places"])
    if len(snippets) > 0:
        return snippets
    else:
        return "No search result found"
    
    
def get_extensions():
    if not SERPAPI_API_KEY:
        print("WARNING: Google Patents extensions are disabled, please set SERPAPI_API_KEY")
        return []

    return [
        ChatbotExtension(
            name="SearchInGooglePatents",
            description="Search in google patents, returning simplified results.",
            execute=search_google_patent,
        ),
        ChatbotExtension(
            name="ReadAGooglePatent",
            description="Read detailed information about a patent from google patents, returning a simplified view.",
            execute=read_a_google_patent,
        )
    ]

async def main():
    res = await search_google_patent(GooglePatentSearchParameters(query="Scheduling Strategy for Enhanced Mobile Broadband (eMBB) and Ultra-Reliable Low-Late"))
    read_res = await read_a_google_patent(GooglePatentReadParameters(patent_id=res[0]['patent_id']))
    # res = await read_patent('https://serpapi.com/search.json?engine=google_patents_details&patent_id=patent%2FTW202135580A%2Fen')
        
if __name__ == "__main__":
    asyncio.run(main())