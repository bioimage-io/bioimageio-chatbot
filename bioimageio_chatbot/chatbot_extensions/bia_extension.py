import httpx
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from bioimageio_chatbot.utils import ChatbotExtension
from schema_agents import schema_tool

class BioImageArchiveClient:
    def __init__(self):
        self._base_url = "https://www.ebi.ac.uk/biostudies/api/v1"

    async def search_bioimage_archive(self, 
        query: str = Field(..., description="The search query string."),
        pageSize: int = Field(10, gt=0, description="Number of search results per page."),
        page: int = Field(1, description="Page number of the search results."),
        sortOrder: Optional[str] = Field("descending", description="Sort order: ascending or descending.")
    ) -> Dict[str, Any]:
        """Search the BioImage Archive for studies, returning simplified search results.  The link format to each study in the results is: https://www.ebi.ac.uk/biostudies/bioimages/studies/{accession}."""
        url = f"{self._base_url}/bioimages/search"
        params = {
            "query": query,
            "pageSize": pageSize,
            "page": page,
            "sortOrder": sortOrder
        }
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
        response.raise_for_status()
        return self._simplify_search_results(response.json())

    def _simplify_search_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        simplified_results = {
            "hits": [
                {
                    "title": hit["title"],
                    "author": hit["author"],
                    "content": hit["content"],
                    "accession": hit["accession"]
                } for hit in results.get("hits", [])
            ],
            "totalHits": results.get("totalHits"),
            "page": results.get("page"),
            "pageSize": results.get("pageSize")
        }
        return simplified_results

    async def read_bioimage_archive_study(self, accession: str = Field(..., description="Accession number of the study.")) -> Dict[str, Any]:
        """Read detailed information about a specific study from the BioImage Archive, returning a simplified dictionary. The link format to the study is: https://www.ebi.ac.uk/biostudies/bioimages/studies/{accession}."""
        url = f"{self._base_url}/studies/{accession}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
        response.raise_for_status()
        return self._simplify_study_details(response.json())

    def _simplify_study_details(self, study_details: Dict[str, Any]) -> Dict[str, Any]:
        # Initialize simplified details with placeholders for title and description
        simplified_details = {
            "title": "",
            "description": "",
            "accession": study_details.get("accno", ""),
            "link": f"https://www.ebi.ac.uk/biostudies/bioimages/studies/{study_details.get('accno', '')}",
            "authors": []
        }

        # Extract title and description from the attributes array by name
        for attribute in study_details.get("section", {}).get("attributes", []):
            if attribute.get("name") == "Title":
                simplified_details["title"] = attribute.get("value", "")
            elif attribute.get("name") == "Description":
                simplified_details["description"] = attribute.get("value", "")

        # Extracting author information
        author_subsections = [sub for sub in study_details.get("section", {}).get("subsections", []) if sub.get("type") == "Author"]
        for author in author_subsections:
            author_attributes = {attr["name"]: attr["value"] for attr in author.get("attributes", [])}
            simplified_details["authors"].append(author_attributes.get("Name", ""))

        return simplified_details



def get_extension():
    bioimage_archive_client = BioImageArchiveClient()
    search_tool = schema_tool(bioimage_archive_client.search_bioimage_archive)
    read_tool = schema_tool(bioimage_archive_client.read_bioimage_archive_study)

    async def get_schema():
        return {
            "search": search_tool.input_model.schema(),
            "read": read_tool.input_model.schema(),
        }

    return ChatbotExtension(
        id="bioimage_archive",
        name="Search BioImage Archive",
        description="A service to search and read studies from the BioImage Archive.",
        get_schema=get_schema, # This is optional, exists only for testing purposes
        tools=dict(
            search=search_tool,
            read=read_tool
        )
    )

if __name__ == "__main__":
    import asyncio
    async def main():
        bioimage_archive_client = BioImageArchiveClient()
        # Example to search in BioImage Archive with simplified results
        search_results = await bioimage_archive_client.search_bioimage_archive(query="cells", pageSize=1)
        print(search_results)

        # Example to read a specific study from BioImage Archive with simplified details
        study_details = await bioimage_archive_client.read_bioimage_archive_study(accession="S-BSST314")
        print(study_details)

    # Run the async function
    asyncio.run(main())
