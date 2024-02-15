import os
import time
import base64
import re
import json
import asyncio

import openai
from pydantic import BaseModel
from openai.types.beta.threads import MessageContentImageFile
from bioimageio_chatbot.build_assistant import load_extensions

import asyncio
import os
import json
import datetime
import secrets
import aiofiles
from imjoy_rpc.hypha import login, connect_to_server

from pydantic import BaseModel, Field
from typing import List, Optional
import pkg_resources
from bioimageio_chatbot.chatbot_extensions import get_builtin_extensions
import logging

logger = logging.getLogger('bioimageio-chatbot')


api_key = os.environ.get("OPENAI_API_KEY")
client = openai.AsyncOpenAI(api_key=api_key)
assistant_id = os.environ.get("ASSISTANT_ID")
instructions = os.environ.get("RUN_INSTRUCTIONS", "")
assistant_title = os.environ.get("ASSISTANT_TITLE", "BioImage.IO Chatbot")
enabled_file_upload_message = os.environ.get("ENABLED_FILE_UPLOAD_MESSAGE", "Upload a file")

#https://github.com/ryo-ma/gpt-assistants-api-ui

async def create_thread(content=None, file=None):
    if content is not None:
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
    else:
        messages = []
    if file is not None:
        messages[0].update({"file_ids": [file.id]})
    thread = await client.beta.threads.create(messages=messages)
    return thread


async def create_message(thread, content, file):
    file_ids = []
    if file is not None:
        file_ids.append(file.id)
    await client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=content, file_ids=file_ids
    )


async def create_run(thread):
    run = await client.beta.threads.runs.create(
        thread_id=thread.id, assistant_id=assistant_id, instructions=instructions
    )
    return run


async def create_file_link(file_name, file_id):
    content = await client.files.content(file_id)
    content_type = content.response.headers["content-type"]
    b64 = base64.b64encode(content.text.encode(content.encoding)).decode()
    link_tag = f'<a href="data:{content_type};base64,{b64}" download="{file_name}">Download Link</a>'
    return link_tag


async def get_message_value_list(messages):
    messages_value_list = []
    for message in messages:
        message_content = ""
        if not isinstance(message, MessageContentImageFile):
            message_content = message.content[0].text
            annotations = message_content.annotations
        else:
            image_file = await client.files.retrieve(message.file_id)
            messages_value_list.append(
                f"Click <here> to download {image_file.filename}"
            )
        citations = []
        for index, annotation in enumerate(annotations):
            message_content.value = message_content.value.replace(
                annotation.text, f" [{index}]"
            )

            if file_citation := getattr(annotation, "file_citation", None):
                cited_file = await client.files.retrieve(file_citation.file_id)
                citations.append(
                    f"[{index}] {file_citation.quote} from {cited_file.filename}"
                )
            elif file_path := getattr(annotation, "file_path", None):
                link_tag = await create_file_link(
                    annotation.text.split("/")[-1], file_path.file_id
                )
                message_content.value = re.sub(
                    r"\[(.*?)\]\s*\(\s*(.*?)\s*\)", link_tag, message_content.value
                )

        message_content.value += "\n" + "\n".join(citations)
        messages_value_list.append(message_content.value)
        return messages_value_list


async def get_message_list(thread, run):
    completed = False
    while not completed:
        run = await client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        print("run.status:", run.status)
        messages = await client.beta.threads.messages.list(thread_id=thread.id)
        messages = await get_message_value_list(messages.data)
        if messages:
            print("messages:", "\n".join(messages))
        if run.status == "completed":
            completed = True
        elif run.status == "failed":
            break
        else:
            time.sleep(0.3)

    messages = await client.beta.threads.messages.list(thread_id=thread.id)
    return await get_message_value_list(messages.data)


class ResponseStep(BaseModel):
    """Response step"""
    name: str = Field(description="Step name")
    details: Optional[dict] = Field(None, description="Step details")

class RichResponse(BaseModel):
    """Rich response with text and intermediate steps"""
    text: str = Field(description="Response text")
    steps: List[ResponseStep] = Field(description="Intermediate steps")

async def get_response(user_input, file, thread, extension_services, status_callback):
    await create_message(thread, user_input, file)
    run = await create_run(thread)
    run = await client.beta.threads.runs.retrieve(
        thread_id=thread.id, run_id=run.id
    )
    steps = []
    while run.status == "in_progress":
        await status_callback({"type": "text", "content": "In progress..."})
        print("run.status:", run.status)
        time.sleep(0.3)
        run = await client.beta.threads.runs.retrieve(
            thread_id=thread.id, run_id=run.id
        )
        run_steps = await client.beta.threads.runs.steps.list(
            thread_id=thread.id, run_id=run.id
        )
        print("run_steps:", run_steps)
        for step in run_steps.data:
            if hasattr(step.step_details, "tool_calls"):
                await status_callback({"type": "text", "content": "tool calls..."})
                for tool_call in step.step_details.tool_calls:
                    if (
                        hasattr(tool_call, "code_interpreter")
                        and tool_call.code_interpreter.input != ""
                    ):
                        input_code = f"### code interpreter\ninput:\n```python\n{tool_call.code_interpreter.input}\n```"
                        print(input_code)
                        steps.append(ResponseStep(name="code interpreter", details={"code": input_code}))

    if run.status == "requires_action":
        print("run.status:", run.status)
        tool_calls = run.required_action.submit_tool_outputs.tool_calls
        tools = [tool_call.function.name for tool_call in tool_calls]
        await status_callback({"type": "text", "content": "Running chatbot extension: " + ", ".join(tools)})
        steps.append(ResponseStep(name="start tool calls", details={"tool_calls": str(run.required_action.submit_tool_outputs.tool_calls)}))
        run, tool_outputs = await execute_action(run, thread, extension_services)
        steps.append(ResponseStep(name="tool call completed", details=tool_outputs))

    await status_callback({"type": "text", "content": "Finishing..."})
    messages = await get_message_list(thread, run)
    return RichResponse(text="\n".join(messages), steps=steps)


async def execute_action(run, thread, extension_services):
    tool_outputs = []
    for tool_call in run.required_action.submit_tool_outputs.tool_calls:
        tool_id = tool_call.id
        tool_function_name = tool_call.function.name
        print(tool_call.function.arguments)

        tool_function_arguments = json.loads(tool_call.function.arguments)

        print("id:", tool_id)
        print("name:", tool_function_name)
        print("arguments:", tool_function_arguments)

        tool_function_output = extension_services[tool_function_name](**tool_function_arguments)
        # check if tool_function_output is awaitable
        if asyncio.iscoroutine(tool_function_output):
            tool_function_output = await tool_function_output


        if isinstance(tool_function_output, BaseModel):
            tool_function_output = tool_function_output.json()
        elif not isinstance(tool_function_output, str):
            tool_function_output = json.dumps(tool_function_output)

        assert isinstance(tool_function_output, str), f"tool_function_output is not a string: {tool_function_output}"
        print("tool_function_output", tool_function_output)
        tool_outputs.append({"tool_call_id": tool_id, "output": tool_function_output})

    run = await client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread.id,
        run_id=run.id,
        tool_outputs=tool_outputs,
    )
    return run, tool_outputs

async def handle_uploaded_file(uploaded_file):
    file = await client.files.create(file=uploaded_file, purpose="assistants")
    return file

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
    
    extension_services = await load_extensions()
    thread = await create_thread()
    
    async def chat(text, chat_history, user_profile=None, status_callback=None, session_id=None, extensions=None, context=None):
        if login_required and context and context.get("user"):
            assert check_permission(context.get("user")), "You don't have permission to use the chatbot, please sign up and wait for approval"
        session_id = session_id or secrets.token_hex(8)
        response = await get_response(text, None, thread, extension_services, status_callback)
        print(f"\nUser: {text}\nChatbot: {response.text}")

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
