from bioimageio_chatbot.utils import ChatbotExtension
from schema_agents import schema_tool
from pydantic import Field, BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
from pathlib import Path
import requests
from markdownify import markdownify as md
import re
import os
from bioimageio_chatbot.utils import download_file
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


EUROBIOIMAGING_TOKEN = os.getenv("EUROBIOIMAGING_TOKEN")
response = requests.get(f"https://www.eurobioimaging.eu/api-general-data.php?a={EUROBIOIMAGING_TOKEN}")
collections = response.json()

for name in ["technologies", "nodes"]:
    for item in collections[name]:
        print(f"Download description from {item['url']}")
        response = requests.get(f"https://www.eurobioimaging.eu/api-page-content.php?a={EUROBIOIMAGING_TOKEN}&url={item['url']}")
        content = md(response.text, heading_style="ATX")
        item["description"] = re.sub(r"\n{3,}", "\n\n", content).strip()

node_index = {}
for item in collections["nodes"]:
    node_index[item['node_id']] = item
    

@schema_tool
async def search_technology(
    keywords: List[str] = Field(..., description="A list of keywords for searching the technology"),
    top_k: int = Field(..., description="Return the top_k number of search results")
):
    """Search Technology in EuroBioImaging service index"""
    results = []
    for item in collections["technologies"]:
        for k in keywords:
            if k.lower() in item["description"].lower() or k.lower() in item["name"].lower():
                results.append(item)
                # TODO sort by relevance
                if len(results) >= top_k:
                    break
        if len(results) >= top_k:
            break
    return results


@schema_tool
async def get_node_details(
    node_id: str = Field(..., description="The EuroBioImaging node id"),
):
    """Get details of the EuroBioImaging node who provide services and technoliges to users"""
    if node_id in node_index:
        return node_index[node_id]
    else:
        return f"Node not found: {node_id}"

@schema_tool
async def search_node(
    keywords: List[str] = Field(..., description="A list of keywords for searching the service nodes"),
    top_k: int = Field(..., description="Return the top_k number of search results")
):
    """Search a service node in the EuroBioImaging network"""
    results = []
    for item in collections["nodes"]:
        for k in keywords:
            if k.lower() in item["description"].lower() or k.lower() in item["name"].lower():
                results.append(item)
                # TODO sort by relevance
                if len(results) >= top_k:
                    break
        if len(results) >= top_k:
            break
    return results


def get_extension():
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