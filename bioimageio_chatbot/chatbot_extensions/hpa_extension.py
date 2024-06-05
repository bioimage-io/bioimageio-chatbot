from bioimageio_chatbot.utils import ChatbotExtension
from schema_agents import schema_tool
from pydantic import Field, BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
from pathlib import Path
import requests
import re
import os
from bioimageio_chatbot.utils import download_file

class HPAClient:
    def __init__(self):
        self._base_url = 'https://www.proteinatlas.org/download/proteinatlas.tsv.zip'
        folder = Path('./data')
        file_path = os.path.join(folder, 'proteinatlas.tsv.zip')
        # firstly check if the data is already downloaded in the /data folder
        if not os.path.exists(file_path):
            os.makedirs(folder, exist_ok=True)
            # download the data
            download_file(self._base_url, file_path)
        # Load and preprocess data at startup
        self.data = pd.read_csv(file_path, delimiter='\t')
        # Convert all textual data to lowercase strings for faster case-insensitive searching
        self.preprocessed_data = self.data.apply(lambda x: x.astype(str).str.lower())
    
    async def search_hpa(self,
        query: str = Field(..., description="Enter gene names, functions, or disease terms to search in the Human Protein Atlas."),
        limitSize: int = Field(10, gt=0, description="Number of returned items per search.")
    ) -> Dict[str, Any]:
        """Search the Human Protein Atlas for proteins based on a query string, return the top search results."""
        query = query.lower()
        
        # Search for the query in the preprocessed data
        query_results = self.preprocessed_data.apply(lambda x: x.str.contains(query)).sum(axis=1)
        query_results = query_results.sort_values(ascending=False)
        query_results = query_results.head(limitSize)

        selected_columns = ['Gene', 'Gene synonym', 'Ensembl', 
                            'Gene description', 'Subcellular location', 'Subcellular main location', 'Subcellular additional location', 
                            'Biological process', 'Molecular function', 'Uniprot', 'Antibody',
                            'Disease involvement', 'Secretome function', 'CCD Protein', 'CCD Transcript',
                            'Evidence', 'Protein class']
        
        info_list = []
        for index in query_results.index:
            items = self.data.loc[index, selected_columns]
            info_list.append(items.to_dict())
        return info_list

    async def read_protein_info(self,
        ensembl: str = Field(..., description="Ensembl ID of the protein.")
    )-> Dict[str, Any]:
        """Get detailed information about a protein from the Human Protein Atlas."""
        json_link = f"https://www.proteinatlas.org/{ensembl}.json"
        response = requests.get(json_link)
        # check if the request was successful
        response.raise_for_status()
        # return the content
        return response.json()



    async def get_cell_image(self,
        gene: str = Field(..., description="Gene name of the protein."),
        ensembl: str = Field(..., description="Ensembl ID of the protein."),
        section: str = Field("subcellular", description="Section of the Human Protein Atlas to search for the protein. Valid options are 'subcellular', 'tissue',")
        ) -> List[str]:
        """Retrieve a list of cell image links from the Human Protein Atlas, where a specific protein is tagged in the green channel. 
        ALWAYS render the result thumbnail images as a horizatal table and create link (format: `[![](http://..._thumb.jpg)](http://....jpg)`) to the full-size image without the '_thumb' suffix."""
        link_name = f"{ensembl}-{gene}"
        http_link = f"https://www.proteinatlas.org/{link_name}/{section}"
        # read the source code of the page
        response = requests.get(http_link)
        if '<p>Not available</p>' in response.text:
            return 'No cell image available.'
        # Search for image links, capturing the part after 'src="'
        pattern = r'src="(?P<url>//images\.proteinatlas\.org/.*?_red_green_thumb\.jpg)"'
        image_links = re.findall(pattern, response.text)
        # replace the 'red_green' with 'blue_red_green_yellow' if 'blue' not in the link, otherwise replace 'blue_red_green' with 'blue_red_green_yellow'
        image_links = [link.replace('red_green', 'blue_red_green_yellow') if 'blue' not in link else link.replace('blue_red_green', 'blue_red_green_yellow') for link in image_links]
        # Remove '_thumb' from each link and print or process them
        final_image_links = []
        for link in image_links:
            final_image_links.append(f"https:{link}")
        return final_image_links


def get_extension():
    hpa_client = HPAClient()
    search_tool = schema_tool(hpa_client.search_hpa)
    read_tool = schema_tool(hpa_client.read_protein_info)
    get_cell_image_tool = schema_tool(hpa_client.get_cell_image)

    return ChatbotExtension(
        id="hpa",
        name="Human Protein Atlas",
        description="Search the Human Protein Atlas to find human protein-related information, including gene expressions, functions, locations, disease associations, and cell images etc. When searching for cell images, always search for the gene name and Ensembl ID of the protein.",
        tools=dict(
            search=search_tool,
            read=read_tool,
            get_cell_image=get_cell_image_tool
        )
    )

if __name__ == "__main__":
    import asyncio
    async def main():
        extension = get_extension()
        query = "brain"
        limitSize = 2
        print(await extension.tools["search"](query=query, limitSize=limitSize))
        # test only one image
        # print(await extension.tools["inspect"](images=[ImageInfo(url="https://bioimage.io/static/img/bioimage-io-icon.png", title="BioImage.io Icon")], query="What is this?", context_description="Inspect the BioImage.io icon."))
    # Run the async function
    asyncio.run(main())
