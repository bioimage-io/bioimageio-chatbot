import asyncio
import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel, Field
from bioimageio_chatbot.utils import ChatbotExtension


class BiiiQuery(BaseModel):
    """Queries parameters for biii.eu search"""

    queries: List[str] = Field(description="A list of keywords to search for")


class BiiiRow(BaseModel):
    """Search result row from biii.eu"""

    name: str = Field(description="Name")
    relevance: str = Field(description="Relevance score")
    image_dimension: Optional[str] = Field(
        None, description="Supported image dimension"
    )
    requires: Optional[str] = Field(description="Dependent software")
    excerpt: str = Field(description="Description")


def extract_table_with_links(table, base_url) -> pd.DataFrame:
    """
    Extracts a table from HTML and includes hyperlinks in the cells if available.

    Args:
    table (bs4.element.Tag): A BeautifulSoup Tag object representing a table.

    Returns:
    pd.DataFrame: A DataFrame representation of the table with text and hyperlinks.
    """
    rows = table.find_all("tr")
    data = []

    for index, row in enumerate(rows):
        columns = row.find_all(["td", "th"])
        row_data = []

        for column in columns:
            cell_text = column.get_text(strip=True)

            # Check for a hyperlink in the cell
            link = column.find("a", href=True)
            if index != 0 and link and cell_text:
                cell_text += f"({link['href'] if link['href'].startswith('http') else base_url + link['href']})"

            row_data.append(cell_text)

        data.append(row_data)

    if data:
        columns = data[0]
        columns[0] = "Name"
        if columns[3] == "Supported Image Dimension":
            columns[2] = "Logo"
    df = pd.DataFrame(data[1:], columns=columns) if data and columns else pd.DataFrame()
    # remove column named "Content type" if exists
    if "Content type" in df.columns:
        df = df.drop(columns=["Content type"])

    # convert to list of BiiiRow
    df = df.to_dict(orient="records")
    return [
        BiiiRow(
            name=row["Name"],
            relevance=row["Relevance"],
            image_dimension=row.get("Supported Image Dimension"),
            requires=row.get("Requires"),
            excerpt=row["Excerpt"],
        )
        for row in df
    ]


def search_biii_with_links(
    queries: List[str], content_type="software", base_url="https://biii.eu"
) -> dict:
    """
    Modified search function to include hyperlinks in the extracted tables.

    Args:
    queries (List[str]): A list of search queries.

    Returns:
    dict: A dictionary where each key is a "Content type" and value is a pandas dataframe of the table with links.
    """
    search_base_url = "https://biii.eu/search?search_api_fulltext="

    for query in queries:
        url = search_base_url + ",".join(query.split())
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        tables = soup.find_all("table")

        for table in tables:
            caption = table.find("caption")
            if caption:
                caption = (
                    caption.get_text().strip().replace("Content type: ", "").lower()
                )
            else:
                continue  # Skip tables without a caption

            if caption != content_type:
                continue

            df = extract_table_with_links(table, base_url)
            return df


class SearchBiii(BaseModel):
    """Find bioimage analysis tools on BioImage Informatics Index (biii.eu). It contains most basic information about image analysis tools."""
    keywords: List[str] = Field(
        description="A list of search keywords, no space allowed in each keyword."
    )
    top_k: int = Field(
        10,
        description="The maximum number of search results to return. Should use a small number to avoid overwhelming the user.",
    )


class BiiiSearchResult(BaseModel):
    """Search results from biii.eu"""
    results: List[BiiiRow] = Field(description="Search results from biii.eu")
    base_url: str = Field(
        description="The based URL of the search results, e.g. ImageJ (/imagej) will become <base_url>/imagej"
    )


class BiiiResponse(BaseModel):
    """Summarize the search results from biii.eu"""

    response: str = Field(
        description="The answer to the user's question based on the search results. Can be either a detailed response in markdown format if the search results are relevant to the user's question or 'I don't know'. It should resolve relative URLs in the search results using the base_url."
    )


async def run_extension(req: SearchBiii):
    # limit top_k from 1 to 15
    req.top_k = max(1, min(req.top_k, 15))
    print(f"Searching biii.eu with keywords: {req.keywords}, top_k: {req.top_k}")
    loop = asyncio.get_running_loop()
    # steps.append(ResponseStep(name="Search on biii.eu", details=req.dict()))
    results = await loop.run_in_executor(
        None, search_biii_with_links, req.keywords, "software", ""
    )
    if results:
        results = BiiiSearchResult(
            results=results[: req.top_k],
            base_url="https://biii.eu",
        )
        return results
    else:
        return f"Sorry I didn't find relevant information in biii.eu about {req.keywords}"

def get_extensions():
    return [
        ChatbotExtension(
            name="SearchInBiii",
            description="Search software tools on BioImage Informatics Index (biii.eu) is a platform for sharing bioimage analysis software and tools.",
            execute=run_extension,
        )
    ]


if __name__ == "__main__":
    results = search_biii_with_links(["image segmentation"])
    # Index(['Name', 'Relevance', 'Logo', 'Supported Image Dimension', 'requires',
    #   'Content type', 'Excerpt'],
    #  dtype='object')
    print(results)
