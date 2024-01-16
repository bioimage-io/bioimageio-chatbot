import asyncio
import os
import json
import datetime
import secrets
import aiofiles
import traceback
from imjoy_rpc.hypha import login, connect_to_server

from pydantic import BaseModel, Field
from schema_agents.role import Role
from schema_agents.schema import Message
from typing import Any, Dict, List, Optional
import pkg_resources
from bioimageio_chatbot.jsonschema_pydantic import jsonschema_to_pydantic, JsonSchemaObject
from bioimageio_chatbot.chatbot_extensions import get_builtin_extensions
from bioimageio_chatbot.utils import extract_schemas, ChatbotExtension
import logging

logger = logging.getLogger('bioimageio-chatbot')

GENERIC_RESPONSE_PROMPT = """Create direct response for various scenarios:
- General Inquiries: When faced with straightforward questions where a direct answer is possible, the chatbot should provide clear, accurate, and concise responses.
- Educational Support: If a question is related to learning or seeks explanation about specific concepts or terms, the chatbot's response should be informative and educational, aiding in the user's understanding of the topic.
- Technical Assistance: For queries that involve technical aspects, such as scripting, coding, or specific professional knowledge, the chatbot should offer detailed, practical advice or solutions. This includes providing example codes, step-by-step guides, or explanations tailored to the user's level of expertise.
- Coding Assistance: If a question is related to coding, the chatbot should provide detailed, practical advice or solutions. This includes providing valid source code or script with comments tailored to the user's level of expertise.
For other types of queries, set use_tools to True to use tools to obtain additional information to help the user.
"""

class UserProfile(BaseModel):
    """The user's profile. This will be used to personalize the response to the user."""
    name: str = Field(description="The user's name.", max_length=32)
    occupation: str = Field(description="The user's occupation.", max_length=128)
    background: str = Field(description="The user's background.", max_length=256)

class ExtensionCallInput(BaseModel):
    """Result of calling an extension function"""
    user_question: str = Field(description="The user's original question.")
    user_profile: Optional[UserProfile] = Field(None, description="The user's profile. You should use this to personalize the response.")
    results: Optional[List[Any]] = Field(None, description="The results of calling extensions.")

class DocumentResponse(BaseModel):
    """The Document Response to the user's question based on the preliminary response and the documentation search results. The response should be tailored to uer's info if provided. 
    If the documentation search results are relevant to the user's question, provide a text response to the question based on the search results.
    If the documentation search results contains only low relevance scores or if the question isn't relevant to the search results, return the preliminary response.
    Importantly, if you can't confidently provide a relevant response to the user's question, return 'Sorry I didn't find relevant information, please try again.'."""
    response: str = Field(description="The answer to the user's question based on the search results. Can be either a detailed response in markdown format if the search results are relevant to the user's question or 'I don't know'.")

class ExtensionCallResponse(BaseModel):
    """Summarize the result of calling an extension function.
    If the function call results are irrelevant to the user's question or if the extension function fails, tell the user that you don't know the answer and provide a summary fo the results.
    """
    relevant: bool = Field(description="Whether the results of calling the extension function are relevant to the user's question.")
    response: str = Field(description="The answer to the user's question based on the result of calling the extension function.")

class QuestionWithHistory(BaseModel):
    """The user's question, chat history, and user's profile."""
    question: str = Field(description="The user's question.")
    chat_history: Optional[List[Dict[str, str]]] = Field(None, description="The chat history.")
    user_profile: Optional[UserProfile] = Field(None, description="The user's profile. You should use this to personalize the response based on the user's background and occupation.")
    chatbot_extensions: Optional[List[Dict[str, Any]]] = Field(None, description="Chatbot extensions.")

class ResponseStep(BaseModel):
    """Response step"""
    name: str = Field(description="Step name")
    details: Optional[dict] = Field(None, description="Step details")

class RichResponse(BaseModel):
    """Rich response with text and intermediate steps"""
    text: str = Field(description="Response text")
    steps: List[ResponseStep] = Field(description="Intermediate steps")


def create_customer_service(builtin_extensions):
    async def respond_to_user(question_with_history: QuestionWithHistory = None, role: Role = None) -> str:
        """Answer the user's question directly or retrieve relevant documents from the documentation, or create a Python Script to get information about details of models."""
        steps = []
        inputs = [question_with_history.user_profile] + list(question_with_history.chat_history) + [question_with_history.question]
        available_tools_prompt = "\n".join([f"- {ext.name}: {ext.description}" for ext in builtin_extensions])
        class GenericResponse(BaseModel):
            """Create response to anwser user's question"""
            use_tools: bool = Field(description="Whether to use tools to answer the user's question. The available tools are:\n" + available_tools_prompt)
            response: Optional[str] = Field(None, description=GENERIC_RESPONSE_PROMPT+"\nSet to null if use_tools=True.")

        resp = await role.aask(inputs, GenericResponse)
        steps.append(ResponseStep(name="GenericResponse", details=resp.dict()))
        if not resp.use_tools:
            return RichResponse(text=resp.response, steps=steps)
        else:
            assert question_with_history.chatbot_extensions is not None
            chatbot_extensions = []
            builtin_ext_names = {ext.name: ext for ext in builtin_extensions}
            for ext in question_with_history.chatbot_extensions:
                if ext['name'] in builtin_ext_names:
                    extension = builtin_ext_names[ext['name']]
                else:
                    extension = ChatbotExtension.parse_obj(ext)
                
                if extension.get_schema:
                    schema = await extension.get_schema()
                    extension.schema_class = jsonschema_to_pydantic(JsonSchemaObject.parse_obj(schema))
                else:
                    input_schemas, _ = extract_schemas(extension.execute)
                    extension.schema_class = input_schemas[0]
                chatbot_extensions.append(extension)

            extension_types = [mode_d.schema_class for mode_d in chatbot_extensions] if question_with_history.chatbot_extensions else []
            logger.info("Response types: %s", extension_types)
            async def run_extension(req):
                assert isinstance(req, tuple(extension_types)), f"Unknown response type: {type(req)}"
                idx = extension_types.index(type(req))
                extension = chatbot_extensions[idx]
                ext_name = extension.name
                steps.append(ResponseStep(name="Execute: " + ext_name, details=req.dict()))
                try:
                    result = await extension.execute(req)
                    steps.append(ResponseStep(name="Output: " + ext_name, details={"result": result}))
                    return result
                except Exception as e:
                    print(f"Failed to run extension {ext_name}, error: {traceback.format_exc()}")
                    steps.append(ResponseStep(name="Error: " + ext_name, details={"Error": traceback.format_exc()}))
                    raise e

            responses = await role.aask(inputs, tuple(extension_types), use_tool_calls=True)
            futs = []
            for resp in responses:
                futs.append(run_extension(resp))

            results = await asyncio.gather(*futs)

            resp = await role.aask(ExtensionCallInput(user_question=question_with_history.question, user_profile=question_with_history.user_profile, results=results), ExtensionCallResponse)
            steps.append(ResponseStep(name="Final Result", details=resp.dict()))
            return RichResponse(text=resp.response, steps=steps)

    customer_service = Role(
        name="Melman",
        profile="Customer Service",
        goal="Your goal as Melman from Madagascar, the community knowledge base manager, is to assist users in effectively utilizing the knowledge base for bioimage analysis. You are responsible for answering user questions, providing clarifications, retrieving relevant documents, and executing scripts as needed. Your overarching objective is to make the user experience both educational and enjoyable.",
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
    builtin_extensions = get_builtin_extensions()
    login_required = os.environ.get("BIOIMAGEIO_LOGIN_REQUIRED") == "true"
    chat_logs_path = os.environ.get("BIOIMAGEIO_CHAT_LOGS_PATH", "./chat_logs")
    assert chat_logs_path is not None, "Please set the BIOIMAGEIO_CHAT_LOGS_PATH environment variable to the path of the chat logs folder."
    if not os.path.exists(chat_logs_path):
        print(f"The chat session folder is not found at {chat_logs_path}, will create one now.")
        os.makedirs(chat_logs_path, exist_ok=True)
    
    customer_service = create_customer_service(builtin_extensions)
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
        # get the chatbot version
        version = pkg_resources.get_distribution('bioimageio-chatbot').version
        chat_his_dict = {'type':user_report['type'],
                         'feedback':user_report['feedback'],
                         'conversations':user_report['messages'], 
                         'session_id':user_report['session_id'], 
                        'timestamp': str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), 
                        'user': context.get('user'),
                        'version': version}
        session_id = user_report['session_id'] + secrets.token_hex(4)
        filename = f"report-{session_id}.json"
        # Create a chat_log.json file inside the session folder
        chat_log_full_path = os.path.join(chat_logs_path, filename)
        await save_chat_history(chat_log_full_path, chat_his_dict)
        print(f"User report saved to {filename}")
        
    async def chat(text, chat_history, user_profile=None, status_callback=None, session_id=None, extensions=None, context=None):
        if login_required and context and context.get("user"):
            assert check_permission(context.get("user")), "You don't have permission to use the chatbot, please sign up and wait for approval"
        session_id = session_id or secrets.token_hex(8)
        # Listen to the `stream` event
        async def stream_callback(message):
            if message.type in ["function_call", "text"]:
                if message.session.id == session_id:
                    await status_callback(message.dict())

        event_bus.on("stream", stream_callback)

        m = QuestionWithHistory(question=text, chat_history=chat_history, user_profile=UserProfile.parse_obj(user_profile), chatbot_extensions=extensions)
        try:
            response = await customer_service.handle(Message(content="", data=m , role="User", session_id=session_id))
            # get the content of the last response
            response = response[-1].data # type: RichResponse
            print(f"\nUser: {text}\nChatbot: {response.text}")
        except Exception as e:
            event_bus.off("stream", stream_callback)
            raise e
        else:
            event_bus.off("stream", stream_callback)

        if session_id:
            chat_history.append({ 'role': 'user', 'content': text })
            chat_history.append({ 'role': 'assistant', 'content': response.text })
            version = pkg_resources.get_distribution('bioimageio-chatbot').version
            chat_his_dict = {'conversations':chat_history, 
                     'timestamp': str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), 
                     'user': context.get('user'),
                     'version': version}
            filename = f"chatlogs-{session_id}.json"
            chat_log_full_path = os.path.join(chat_logs_path, filename)
            await save_chat_history(chat_log_full_path, chat_his_dict)
            print(f"Chat history saved to {filename}")
        return response.dict()

    async def ping(context=None):
        if login_required and context and context.get("user"):
            assert check_permission(context.get("user")), "You don't have permission to use the chatbot, please sign up and wait for approval"
        return "pong"

    def encode_base_model(data):
        return data.dict()

    server.register_codec({
        "name": "pydantic-base-model",
        "type": BaseModel,
        "encoder": encode_base_model,
    })

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
        "builtin_extensions": [{"name": ext.name, "description": ext.description} for ext in builtin_extensions],
    })
    
    version = pkg_resources.get_distribution('bioimageio-chatbot').version
    def reload_index():
        with open(os.path.join(os.path.dirname(__file__), "static/index.html"), "r", encoding="utf-8") as f:
            index_html = f.read()
        index_html = index_html.replace("https://ai.imjoy.io", server.config['public_base_url'] or f"http://127.0.0.1:{server.config['port']}")
        index_html = index_html.replace('"bioimageio-chatbot"', f'"{hypha_service_info["id"]}"')
        index_html = index_html.replace('v0.1.0', f'v{version}')
        index_html = index_html.replace("LOGIN_REQUIRED", "true" if login_required else "false")
        return index_html
    
    index_html = reload_index()
    debug = os.environ.get("BIOIMAGEIO_DEBUG") == "true"
    async def index(event, context=None):
        return {
            "status": 200,
            "headers": {'Content-Type': 'text/html'},
            "body": reload_index() if debug else index_html,
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

    user_url = f"{server_url}/{server.config['workspace']}/apps/bioimageio-chatbot-client/index"
    print(f"The BioImage.IO Chatbot is available at: {user_url}")
    
if __name__ == "__main__":
    server_url = """https://ai.imjoy.io"""
    loop = asyncio.get_event_loop()
    loop.create_task(connect_server(server_url))
    loop.run_forever()