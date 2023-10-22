import asyncio
import os
from imjoy_rpc.hypha import login, connect_to_server

from pydantic import BaseModel, Field
from schema_agents.role import Role
from schema_agents.schema import Message
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from typing import Any, Dict, List, Optional, Union
import requests
import sys
import io
from bioimageio_chatbot.knowledge_base import load_knowledge_base
from bioimageio_chatbot.utils import get_manifest, download_file
import pkg_resources

def load_model_info():
    response = requests.get("https://bioimage-io.github.io/collection-bioimage-io/collection.json")
    assert response.status_code == 200
    model_info = response.json()
    resource_items = model_info['collection']
    return resource_items

def execute_code(script, context=None):
    if context is None:
        context = {}

    # Redirect stdout and stderr to capture their output
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    try:
        # Create a copy of the context to avoid modifying the original
        local_vars = context.copy()

        # Execute the provided Python script with access to context variables
        exec(script, local_vars)

        # Capture the output from stdout and stderr
        stdout_output = sys.stdout.getvalue()
        stderr_output = sys.stderr.getvalue()

        return {
            "stdout": stdout_output,
            "stderr": stderr_output,
            # "context": local_vars  # Include context variables in the result
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            # "context": context  # Include context variables in the result even if an error occurs
        }
    finally:
        # Restore the original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr


class ModelZooInfoScriptResults(BaseModel):
    """Results of executing the model zoo info query script."""
    stdout: str = Field(description="The output from stdout.")
    stderr: str = Field(description="The output from stderr.")
    request: str = Field(description="User's request in details")
    user_info: str = Field(description="User info for personalize response.")

class DirectResponse(BaseModel):
    """Direct response to a user's question."""
    response: str = Field(description="The response to the user's question.")

class DocWithScore(BaseModel):
    """A document with relevance score."""
    doc: str = Field(description="The document retrieved.")
    score: float = Field(description="The relevance score of the document.")
    metadata: Dict[str, Any] = Field(description="The metadata of the document.")

class DocumentSearchInput(BaseModel):
    """Results of document retrieval from documentation."""
    user_question: str = Field(description="The user's original question.")
    relevant_context: List[DocWithScore] = Field(description="Context chunks from the documentation")
    user_info: str = Field(description="User info for personalize response.")
    base_url: Optional[str] = Field(None, description="The base url of the documentation, used for resolve relative URL in the document and produce markdown links.")
    format: Optional[str] = Field(None, description="The format of the document.")

class FinalResponse(BaseModel):
    """The final response to the user's question. If the retrieved context has low relevance score, or the question isn't relevant to the retried context, return 'I don't know'."""
    response: str = Field(description="The answer to the user's question in markdown format.")


class UserProfile(BaseModel):
    """The user's profile. This will be used to personalize the response."""
    name: str = Field(description="The user's name.", max_length=32)
    occupation: str = Field(description="The user's occupation. ", max_length=128)
    background: str = Field(description="The user's background. ", max_length=256)

class QuestionWithHistory(BaseModel):
    """The user's question, chat history and user's profile."""
    question: str = Field(description="The user's question.")
    chat_history: Optional[List[Dict[str, str]]] = Field(None, description="The chat history.")
    user_profile: Optional[UserProfile] = Field(None, description="The user's profile. You should use this to personalize the response based on the user's background and occupation.")
    channel_id: Optional[str] = Field(None, description="The channel id of the user's question. This is used to limit the search scope to a specific channel, None means all the channels.")

def create_customer_service(db_path):
    collections = get_manifest()['collections']
    docs_store_dict = load_knowledge_base(db_path)
    collection_info_dict = {collection['id']: collection for collection in collections}
    resource_items = load_model_info()
    types = set()
    tags = set()
    for resource in resource_items:
        types.add(resource['type'])
        tags.update(resource['tags'])
    types = list(types)
    tags = list(tags)[:10]
    
    channels_info = "\n".join(f"""- `{collection['id']}`: {collection['description']}""" for collection in collections)
    resource_item_stats = f"""Each item contains the following fields: {list(resource_items[0].keys())}\nThe available resource types are: {types}\nSome example tags: {tags}\nHere is an example: {resource_items[0]}"""
    class DocumentRetrievalInput(BaseModel):
        """Input for finding relevant documents from database."""
        query: str = Field(description="Query used to retrieve related documents.")
        request: str = Field(description="User's request in details")
        user_info: str = Field(description="Brief user info summary for personalized response, including name, background etc.")
        database_id: str = Field(description=f"Select a database for information retrieval. The available databases are:\n{channels_info}")

    class ModelZooInfoScript(BaseModel):
        """Create a Python Script to get information about details of models, applications and datasets etc."""
        script: str = Field(description="The script to be executed, the script use a predefined local variable `resources` which contains a list of dictionaries with all the resources in the model zoo (including models, applications, datasets etc.), the response to the query should be printed to the stdout. Details about the `resources`:\n" + resource_item_stats)
        request: str = Field(description="User's request in details")
        user_info: str = Field(description="Brief user info summary for personalized response, including name, background etc.")

    async def respond_to_user(question_with_history: QuestionWithHistory = None, role: Role = None) -> str:
        """Answer the user's question directly or retrieve relevant documents from the documentation, or create a Python Script to get information about details of models."""
        inputs = [question_with_history.user_profile] + list(question_with_history.chat_history) + [question_with_history.question] 
        req = await role.aask(inputs, Union[DirectResponse, DocumentRetrievalInput, ModelZooInfoScript])
        if isinstance(req, DirectResponse):
            return req.response
        elif isinstance(req, DocumentRetrievalInput):
            # Use the automatic channel selection if the user doesn't specify a channel
            selected_channel = question_with_history.channel_id or req.database_id
            docs_store = docs_store_dict[selected_channel]
            collection_info = collection_info_dict[selected_channel]
            results_with_scores = await docs_store.asimilarity_search_with_relevance_scores(req.query, k=3)
            docs_with_score = [DocWithScore(doc=doc.page_content, score=score, metadata=doc.metadata) for doc, score in results_with_scores]
            search_input = DocumentSearchInput(user_question=req.request, relevant_context=docs_with_score, user_info=req.user_info, base_url=collection_info.get('base_url'), format=collection_info.get('format'))
            response = await role.aask(search_input, FinalResponse)
            return response.response
        elif isinstance(req, ModelZooInfoScript):
            loop = asyncio.get_running_loop()
            print(f"Executing the script:\n{req.script}")
            result = await loop.run_in_executor(None, execute_code, req.script, {"resources": resource_items})
            print(f"Script execution result:\n{result}")
            response = await role.aask(ModelZooInfoScriptResults(
                stdout=result["stdout"],
                stderr=result["stderr"],
                request=req.request,
                user_info=req.user_info
            ), FinalResponse)
            return response.response
        
    CustomerServiceRole = Role.create(
        name="Melman",
        profile="Customer Service",
        goal="Your goal as Melman, the community knowledge base manager, is to assist users in effectively utilizing the BioImage.IO knowledge base for bioimage analysis. You are responsible for answering user questions, providing clarifications, retrieving relevant documents, and executing scripts as needed. Your overarching objective is to make the user experience both educational and enjoyable.",
        constraints=None,
        actions=[respond_to_user],
    )
    customer_service = CustomerServiceRole()
    return customer_service

async def connect_server(server_url):
    token = await login({"server_url": server_url})
    server = await connect_to_server({"server_url": server_url, "token": token, "method_timeout": 100})
    await register_chat_service(server)
    
async def register_chat_service(server):
    """Hypha startup function."""
    collections = get_manifest()['collections']
    knowledge_base_path = os.environ.get("BIOIMAGEIO_KNOWLEDGE_BASE_PATH", "./bioimageio-knowledge-base")
    assert knowledge_base_path is not None, "Please set the BIOIMAGEIO_KNOWLEDGE_BASE_PATH environment variable to the path of the knowledge base."
    if not os.path.exists(knowledge_base_path):
        print(f"The knowledge base is not found at {knowledge_base_path}, will download it automatically.")
        os.makedirs(knowledge_base_path, exist_ok=True)

    channel_id_by_name = {collection['name']: collection['id'] for collection in collections}
    customer_service = create_customer_service(knowledge_base_path)

    async def chat(text, chat_history, user_profile=None, channel=None, context=None):
        # Get the channel id by its name
        if channel == 'auto':
            channel = None
        if channel:
            assert channel in channel_id_by_name, f"Channel {channel} is not found, available channels are {list(channel_id_by_name.keys())}"
            channel_id = channel_id_by_name[channel]
        else:
            channel_id = None
        
        # user_profile = {"name": "lulu", "occupation": "data scientist", "background": "machine learning and AI"}
        m = QuestionWithHistory(question=text, chat_history=chat_history, user_profile=UserProfile.parse_obj(user_profile), channel_id=channel_id)
        response = await customer_service.handle(Message(content=m.json(), instruct_content=m , role="User"))
        # get the content of the last response
        response = response[-1].content
        print(f"\nUser: {text}\nBot: {response}")
        return response

    hypha_service_info = await server.register_service({
        "name": "BioImage.IO Chatbot",
        "id": "bioimageio-chatbot",
        "config": {
            "visibility": "public",
            "require_context": True
        },
        "chat": chat,
        "channels": [collection['name'] for collection in collections]
    })
    
    version = pkg_resources.get_distribution('bioimageio-chatbot').version
    
    with open(os.path.join(os.path.dirname(__file__), "static/index.html"), "r") as f:
        index_html = f.read()
    index_html = index_html.replace("https://ai.imjoy.io", server.config['public_base_url'])
    index_html = index_html.replace('"bioimageio-chatbot"', f'"{hypha_service_info["id"]}"')
    index_html = index_html.replace('v0.1.0', f'v{version}')
    async def index(event, context=None):
        return {
            "status": 200,
            "headers": {'Content-Type': 'text/html'},
            "body": index_html
        }
    
    await server.register_service({
        "id": "bioimageio-chatbot-client",
        "type": "functions",
        "config": {
            "visibility": "public",
            "require_context": False
        },
        "index": index,
    })
    server_url = server.config['public_base_url']

    print(f"The BioImage.IO Chatbot is available at: {server_url}/{server.config['workspace']}/apps/bioimageio-chatbot-client/index")

if __name__ == "__main__":
    # asyncio.run(main())
    server_url = "https://ai.imjoy.io"
    loop = asyncio.get_event_loop()
    loop.create_task(connect_server(server_url))
    loop.run_forever()