import asyncio
import os
import json
import datetime
import secrets
import aiofiles
from imjoy_rpc.hypha import login, connect_to_server

from pydantic import BaseModel, Field
from schema_agents.role import Role
from schema_agents.schema import Message
from typing import Any, Dict, List, Optional, Union
import requests
import sys
import io
from bioimageio_chatbot.knowledge_base import load_knowledge_base
from bioimageio_chatbot.utils import get_manifest
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
    user_info: Optional[str] = Field("", description="User info for personalize response.")

class DirectResponse(BaseModel):
    """Direct response to a user's question."""
    response: str = Field(description="The response to the user's question.")

class DocWithScore(BaseModel):
    """A document with relevance score and relevant information."""
    doc: str = Field(description="The document retrieved.")
    score: float = Field(description="The relevance score of the document.")
    metadata: Dict[str, Any] = Field(description="The metadata of the document.")
    base_url: Optional[str] = Field(None, description="The base url of the documentation, used for resolve relative URL in the document and produce markdown links.")
    format: Optional[str] = Field(None, description="The format of the document.")
    
class DocumentSearchInput(BaseModel):
    """Results of document retrieval from documentation."""
    user_question: str = Field(description="The user's original question.")
    relevant_context: List[DocWithScore] = Field(description="Context chunks from the documentation")
    user_info: Optional[str] = Field("", description="User info for personalize response.")
    
class FinalResponse(BaseModel):
    """The final response to the user's question. If the retrieved context has low relevance score, or the question isn't relevant to the retrieved context, return 'I don't know'."""
    response: str = Field(description="The answer to the user's question in markdown format.")

class ChannelInfo(BaseModel):
    """The selected channel of the user's question. If provided, stick to the selected channel when answering the user's question."""
    id: str = Field(description="The channel id.")
    name: str = Field(description="The channel name.")
    description: str = Field(description="The channel description.")

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
    channel_info: Optional[ChannelInfo] = Field(None, description="The selected channel of the user's question. If provided, stick to the selected channel when answering the user's question.")

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
    class ChannelQuery(BaseModel):
        """A query to a specific channel."""
        query: str = Field(description="Query to be used for retrieving relevant documents at specific channel.")
        channel_id: str = Field(description=f"Channel used for retrieving. The available channels are:\n{channels_info}")
        k: int = Field(description="The top-k number of documents to be retrieved from the channel, higher k number means more documents to retriev. Default 2, maximum 20.")
    
    # class DocumentRetrievalInput(BaseModel):
    #     """Retrieve context information from one or more data channels, the number of queries and corresponding channels should be decided based on the user's request. For example, you can create a query for each channel, or different queries for different channels."""
    #     queries: List[ChannelQuery] = Field(description="A list of queries to be used for retrieving relevant documents in one or more channels. The channel_id for each query MUST be the same as the user provided channel_id, and if not specified it should be select based on the query automatically. The available channels are:\n{channels_info}")
    #     request: str = Field(description="User's request in details")
    #     user_info: Optional[str] = Field("",description="Brief user info summary for personalized response, including name, background etc.")
    class DocumentRetrievalInput(BaseModel):
        """
        Request information retrieval from multiple data channels based on user queries.

        """
        queries: List[ChannelQuery] = Field(description="A list of queries to be used for retrieving relevant documents in one or more channels. The channel_id for each query MUST be the same as the user provided channel_id, and if not specified it should be select based on the query automatically. The available channels are:\n{channels_info}")
        request: str = Field(description="User's request in details")
        user_info: Optional[str] = Field("",description="Brief user info summary for personalized response, including name, background etc.")
    
    class ModelZooInfoScript(BaseModel):
        """Create a Python Script to get information about details of models, applications and datasets etc."""
        script: str = Field(description="The script to be executed, the script use a predefined local variable `resources` which contains a list of dictionaries with all the resources in the model zoo (including models, applications, datasets etc.), the response to the query should be printed to the stdout. Details about the `resources`:\n" + resource_item_stats)
        request: str = Field(description="User's request in details")
        user_info: Optional[str] = Field("", description="Brief user info summary for personalized response, including name, background etc.")

    async def respond_to_user(question_with_history: QuestionWithHistory = None, role: Role = None) -> str:
        """Answer the user's question directly or retrieve relevant documents from the documentation, or create a Python Script to get information about details of models. """
        inputs = [question_with_history.user_profile] + list(question_with_history.chat_history) + [question_with_history.question]
        # The channel info will be inserted at the beginning of the inputs
        if question_with_history.channel_info:
            inputs.insert(0, question_with_history.channel_info)
        if not question_with_history.channel_info or question_with_history.channel_info.id == "bioimage.io":
            try:
                req = await role.aask(inputs, Union[DirectResponse, DocumentRetrievalInput, ModelZooInfoScript])
            except Exception as e:
                # try again
                req = await role.aask(inputs, Union[DirectResponse, DocumentRetrievalInput, ModelZooInfoScript])
        else:
            try:
                req = await role.aask(inputs, Union[DirectResponse, DocumentRetrievalInput])
            except Exception as e:
                # try again
                req = await role.aask(inputs, Union[DirectResponse, DocumentRetrievalInput])
        if isinstance(req, DirectResponse):
            return req.response
        elif isinstance(req, DocumentRetrievalInput):
            docs_with_score = []
            # enumerate req.channel_id
            for query_channel in req.queries:
                docs_store = docs_store_dict[query_channel.channel_id]
                collection_info = collection_info_dict[query_channel.channel_id]
                print(f"Retrieving documents from database {query_channel.channel_id} with query: {query_channel.query} and k={query_channel.k}")
                results_with_scores = await docs_store.asimilarity_search_with_relevance_scores(query_channel.query, k=query_channel.k)
                docs_with_score.extend([DocWithScore(doc=doc.page_content, score=round(score,2), metadata=doc.metadata, base_url=collection_info.get('base_url'), format=collection_info.get('format')) for doc, score in results_with_scores])
            for i in range(len(docs_with_score)):
                print(f"Retrieved document:\n{docs_with_score[i].doc[:20] + '...'} (score: {docs_with_score[i].score})")
            search_input = DocumentSearchInput(user_question=req.request, relevant_context=docs_with_score, user_info=req.user_info)
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
        
    customer_service = Role(
        name="Melman",
        profile="Customer Service",
        goal="Your goal as Melman from Madagascar, the community knowledge base manager, is to assist users in effectively utilizing the BioImage.IO knowledge base for bioimage analysis. You are responsible for answering user questions, providing clarifications, retrieving relevant documents, and executing scripts as needed.  Your overarching objective is to make the user experience both educational and enjoyable.",
        constraints=None,
        actions=[respond_to_user],
    )
    return customer_service

async def save_chat_history(chat_log_full_path, chat_his_dict):    
    # Serialize the chat history to a json string
    chat_history_json = json.dumps(chat_his_dict)

    # Write the serialized chat history to the json file
    async with aiofiles.open(chat_log_full_path, mode='w', encoding='utf-8') as f:
        await f.write(chat_history_json)

    
async def connect_server(server_url):
    """Connect to the server and register the chat service."""
    login_required = os.environ.get("BIOIMAGEIO_LOGIN_REQUIRED") == "true"
    if login_required:
        token = await login({"server_url": server_url})
    else:
        token = None
    server = await connect_to_server({"server_url": server_url, "token": token, "method_timeout": 100})
    await register_chat_service(server)
    
async def register_chat_service(server):
    """Hypha startup function."""
    collections = get_manifest()['collections']
    login_required = os.environ.get("BIOIMAGEIO_LOGIN_REQUIRED") == "true"
    knowledge_base_path = os.environ.get("BIOIMAGEIO_KNOWLEDGE_BASE_PATH", "./bioimageio-knowledge-base")
    assert knowledge_base_path is not None, "Please set the BIOIMAGEIO_KNOWLEDGE_BASE_PATH environment variable to the path of the knowledge base."
    if not os.path.exists(knowledge_base_path):
        print(f"The knowledge base is not found at {knowledge_base_path}, will download it automatically.")
        os.makedirs(knowledge_base_path, exist_ok=True)

    chat_logs_path = os.environ.get("BIOIMAGEIO_CHAT_LOGS_PATH", "./chat_logs")
    assert chat_logs_path is not None, "Please set the BIOIMAGEIO_CHAT_LOGS_PATH environment variable to the path of the chat logs folder."
    if not os.path.exists(chat_logs_path):
        print(f"The chat session folder is not found at {chat_logs_path}, will create one now.")
        os.makedirs(chat_logs_path, exist_ok=True)
    
    channel_id_by_name = {collection['name']: collection['id'] for collection in collections}
    description_by_id = {collection['id']: collection['description'] for collection in collections}
    customer_service = create_customer_service(knowledge_base_path)
    
    event_bus = customer_service.get_event_bus()
    event_bus.register_default_events()
        
    def load_authorized_emails():
        if login_required:
            authorized_users_path = os.environ.get("BIOIMAGEIO_AUTHORIZED_USERS_PATH")
            if authorized_users_path:
                assert os.path.exists(authorized_users_path), f"The authorized users file is not found at {authorized_users_path}"
                with open(authorized_users_path, "r") as f:
                    authorized_users = json.load(f)["users"]
                authorized_emails = [user["email"] for user in authorized_users if "email" in user]
            else:
                authorized_emails = None
        else:
            authorized_emails = None
        return authorized_emails

    authorized_emails = load_authorized_emails()
    def check_permission(user):
        if authorized_emails is None or user["email"] in authorized_emails:
            return True
        else:
            return False
        
    async def report(user_report, context=None):
        if login_required and context and context.get("user"):
            assert check_permission(context.get("user")), "You don't have permission to report the chat history."
        chat_his_dict = {'type':user_report['type'],
                         'feedback':user_report['feedback'],
                         'conversations':user_report['messages'], 
                         'session_id':user_report['session_id'], 
                        'timestamp': str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), 
                        'user': context.get('user')}
        session_id = user_report['session_id'] + secrets.token_hex(4)
        filename = f"report-{session_id}.json"
        # Create a chat_log.json file inside the session folder
        chat_log_full_path = os.path.join(chat_logs_path, filename)
        await save_chat_history(chat_log_full_path, chat_his_dict)
        print(f"User report saved to {filename}")
        
    async def chat(text, chat_history, user_profile=None, channel=None, status_callback=None, session_id=None, context=None):
        if login_required and context and context.get("user"):
            assert check_permission(context.get("user")), "You don't have permission to use the chatbot, please sign up and wait for approval"
        session_id = session_id or secrets.token_hex(8)
        # Listen to the `stream` event
        async def stream_callback(message):
            if message["type"] in ["function_call", "text"]:
                if message["session_id"] == session_id:
                    await status_callback(message)

        event_bus.on("stream", stream_callback)
        
        # Get the channel id by its name
        if channel == 'auto':
            channel = None
        if channel:
            assert channel in channel_id_by_name, f"Channel {channel} is not found, available channels are {list(channel_id_by_name.keys())}"
            channel_id = channel_id_by_name[channel]
        else:
            channel_id = None

        channel_info = channel_id and {"id": channel_id, "name": channel, "description": description_by_id[channel_id]}
        if channel_info:
            channel_info = ChannelInfo.parse_obj(channel_info)
        # user_profile = {"name": "lulu", "occupation": "data scientist", "background": "machine learning and AI"}
        m = QuestionWithHistory(question=text, chat_history=chat_history, user_profile=UserProfile.parse_obj(user_profile),channel_info=channel_info)
        try:
            response = await customer_service.handle(Message(content=m.json(), data=m , role="User"), session_id=session_id)
            # get the content of the last response
            response = response[-1].content
            print(f"\nUser: {text}\nChatbot: {response}")
        except Exception as e:
            event_bus.off("stream", stream_callback)
            raise e
        else:
            event_bus.off("stream", stream_callback)

        if session_id:
            chat_history.append({ 'role': 'user', 'content': text })
            chat_history.append({ 'role': 'assistant', 'content': response })
            chat_his_dict = {'conversations':chat_history, 
                     'timestamp': str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), 
                     'user': context.get('user')}
            filename = f"chatlogs-{session_id}.json"
            chat_log_full_path = os.path.join(chat_logs_path, filename)
            await save_chat_history(chat_log_full_path, chat_his_dict)
            print(f"Chat history saved to {filename}")
        return response

    async def ping(context=None):
        if login_required and context and context.get("user"):
            assert check_permission(context.get("user")), "You don't have permission to use the chatbot, please sign up and wait for approval"
        return "pong"

    hypha_service_info = await server.register_service({
        "name": "BioImage.IO Chatbot",
        "id": "bioimageio-chatbot",
        "config": {
            "visibility": "public",
            "require_context": True
        },
        "ping": ping,
        "chat": chat,
        "report": report,
        "channels": [collection['name'] for collection in collections]
    })
    
    version = pkg_resources.get_distribution('bioimageio-chatbot').version
    
    with open(os.path.join(os.path.dirname(__file__), "static/index.html"), "r") as f:
        index_html = f.read()
    index_html = index_html.replace("https://ai.imjoy.io", server.config['public_base_url'] or f"http://127.0.0.1:{server.config['port']}")
    index_html = index_html.replace('"bioimageio-chatbot"', f'"{hypha_service_info["id"]}"')
    index_html = index_html.replace('v0.1.0', f'v{version}')
    index_html = index_html.replace("LOGIN_REQUIRED", "true" if login_required else "false")
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
