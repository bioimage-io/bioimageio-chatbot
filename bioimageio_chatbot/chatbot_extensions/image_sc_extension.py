import httpx
import os
import os
import urllib.parse
import asyncio
import html2text
from pydantic import BaseModel, Field
from bioimageio_chatbot.utils import ChatbotExtension
from typing import List, Dict, Any, Optional

class SearchParameters(BaseModel):
    """Parameters for searching the Image.sc Forum."""
    query: str = Field(..., description="The search query string.")
    top_k: int = Field(..., gt=0, description="Maximum number of search results to return.")
    order: Optional[str] = Field("latest", description="Order of the search results, options: latest, likes, views, latest_topic.")
    status: Optional[str] = Field(None, description="The status filter for the search results, options: solved, unsolved, open, closed.")

class ReadPostsParameters(BaseModel):
    """Parameters for reading posts or topics from the Image.sc Forum."""
    type: str = Field(..., description="type: `post` or `topic`")
    id: int = Field(..., description="topic id")

class DiscourseClient:
    def __init__(self, base_url: str, username: str, api_key: str):
        self._base_url = base_url
        self._username = username
        self._api_key = api_key

    def _build_query_string(self, query: str, order: str, status: str) -> str:
        # Construct the query string with the provided parameters.
        # Note: `urllib.parse.quote` is used to ensure the query is URL encoded.
        query_components = [
            f"q={urllib.parse.quote(query)}",
            f"order:{order}",
        ]
        if status:
            query_components.append(f"status:{status}")
        return " ".join(query_components)

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Api-Username": self._username,
            "Api-Key": self._api_key,
        }

    def _cleanup_search_results(self, results: Dict[str, Any], top_k: int=10) -> Dict[str, Any]:
        cleaned_results = {
            "posts": [
                {"id": post["id"], "topic_id": post["topic_id"], "blurb": post["blurb"]}
                for post in results.get("posts", [])
                if "id" in post and "topic_id" in post and "blurb" in post
            ],
            "topics": [
                {"title": topic["title"], "slug": topic["slug"]}
                for topic in results.get("topics", [])
                if "title" in topic and "slug" in topic
            ]
        }
        cleaned_results["posts"] = cleaned_results["posts"][:top_k]
        cleaned_results["topics"] = cleaned_results["topics"][:top_k]
        return cleaned_results

    async def search_image_sc(self, req: SearchParameters) -> Dict[str, Any]:
        """Search the Image.sc Forum(a forum for scientific image software) for posts and topics."""
        # Prepare headers for authentication
        headers = self._get_headers()

        # Build the query string
        query_string = self._build_query_string(req.query, req.order, req.status)
        
        # Construct the full URL
        url = f"{self._base_url}/search.json?{query_string}"

        # Perform the asynchronous HTTP GET request
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            return self._cleanup_search_results(response.json(), req.top_k)  # Return the JSON response
        else:
            response.raise_for_status()  # Raise an error for bad responses

    async def read_image_sc_posts(self, req: ReadPostsParameters) -> List[str]:
        """Read a single or all the posts in a topic from the Image.sc Forum (a discussion forum for scientific image software)."""
        if req.type == "post":
            return await self.get_post_content(req.id)
        elif req.type == "topic":
            return await self.get_topic_content(req.id)
    
    async def get_topic_content(self, topic_id: int) -> Dict[str, Any]:
        url = f"{self._base_url}/t/{topic_id}.json"
        headers = self._get_headers()
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
        response.raise_for_status()
        topic_data = response.json()

        post_ids = [post['id'] for post in topic_data['post_stream']['posts']]
        messages = await asyncio.gather(*[self.get_post_content(post_id) for post_id in post_ids])
        posts = []
        for msg in messages:
            posts.append(f"{msg['username']}: {html2text.html2text(msg['content'])}")
        return {"posts": posts, "url": f"{self._base_url}/t/{topic_data['slug']}"}

    async def get_post_content(self, post_id: int) -> str:
        url = f"{self._base_url}/posts/{post_id}.json"
        headers = self._get_headers()
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
        response.raise_for_status()
        post_data = response.json()
        return {"username": post_data["username"], "content": post_data["cooked"], "url": f"{self._base_url}/t/{post_data['topic_slug']}"}
    
def get_extensions():
    username = os.environ.get("DISCOURSE_USERNAME")
    api_key = os.environ.get("DISCOURSE_API_KEY")
    if not username or not api_key:
        print("WARNING: Image.sc Forum extensions require DISCOURSE_USERNAME and DISCOURSE_API_KEY environment variables to be set, disabling it for now.")
        return []

    discourse_client = DiscourseClient(base_url="https://forum.image.sc/", username=username, api_key=api_key)
    return [
        ChatbotExtension(
            name="SearchInImageScForum",
            description="Search the Image.sc Forum for posts and topics.",
            execute=discourse_client.search_image_sc,
        ),
        ChatbotExtension(
            name="ReadImageScForumPosts",
            description="Read a single or all the posts in a topic from the Image.sc Forum.",
            execute=discourse_client.read_image_sc_posts,
        )
    ]

if __name__ == "__main__":
    async def main():
        discourse_client = DiscourseClient(base_url="https://forum.image.sc", username="oeway", api_key="1b8819f9f95bc7f4eb51d3f9bac6d4dd0245569314a7801f670c1067d06c8268")
        # results = await discourse_client.search_image_sc("python ", "latest", 5, "open")
        # print(json.dumps(results))
        results = await discourse_client.read_image_sc_posts('topic', 44826)
        print(results)

    # Run the async function
    asyncio.run(main())
