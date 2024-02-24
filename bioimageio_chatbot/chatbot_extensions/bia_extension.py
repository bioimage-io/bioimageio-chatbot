import httpx
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from bioimageio_chatbot.utils import ChatbotExtension

class BioImageSearchParameters(BaseModel):
    """Parameters for searching the BioImage Archive."""
    query: str = Field(..., description="The search query string.")
    pageSize: int = Field(10, gt=0, description="Number of search results per page.")
    page: int = Field(1, description="Page number of the search results.")
    sortOrder: Optional[str] = Field("descending", description="Sort order: ascending or descending.")

class BioImageReadParameters(BaseModel):
    """Parameters for reading detailed information about a study from the BioImage Archive."""
    accession: str = Field(..., description="Accession number of the study.")

class BioImageArchiveClient:
    def __init__(self):
        self._base_url = "https://www.ebi.ac.uk/biostudies/api/v1"

    async def search_bioimage_archive(self, req: BioImageSearchParameters) -> Dict[str, Any]:
        """Search the BioImage Archive for studies, returning simplified search results.  The link format to each study in the results is: https://www.ebi.ac.uk/biostudies/bioimages/studies/{accession}."""
        url = f"{self._base_url}/bioimages/search"
        params = req.dict(exclude_none=True)
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

    async def read_bioimage_archive_study(self, req: BioImageReadParameters) -> Dict[str, Any]:
        """Read detailed information about a specific study from the BioImage Archive, returning a simplified dictionary. The link format to the study is: https://www.ebi.ac.uk/biostudies/bioimages/studies/{accession}."""
        url = f"{self._base_url}/studies/{req.accession}"
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



def get_extensions():
    bioimage_archive_client = BioImageArchiveClient()
    return [
        ChatbotExtension(
            name="SearchInBioImageArchive",
            description="Search the BioImage Archive for studies, returning simplified results.",
            execute=bioimage_archive_client.search_bioimage_archive,
        ),
        ChatbotExtension(
            name="ReadBioImageArchiveStudy",
            description="Read detailed information about a study from the BioImage Archive, returning a simplified view.",
            execute=bioimage_archive_client.read_bioimage_archive_study,
        )
    ]

if __name__ == "__main__":
    import asyncio
    async def main():
        bioimage_archive_client = BioImageArchiveClient()
        # Example to search in BioImage Archive with simplified results
        search_params = BioImageSearchParameters(query="cells", pageSize=1)
        search_results = await bioimage_archive_client.search_bioimage_archive(search_params)
        print(search_results)

        # Example to read a specific study from BioImage Archive with simplified details
        read_params = BioImageReadParameters(accession="S-BSST314")
        study_details = await bioimage_archive_client.read_bioimage_archive_study(read_params)
        print(study_details)

    # Run the async function
    asyncio.run(main())
