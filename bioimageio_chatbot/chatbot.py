import asyncio
import os
import json
import datetime
import secrets
import aiofiles
from functools import partial
from imjoy_rpc.hypha import login, connect_to_server

from pydantic import BaseModel, Field
from schema_agents import Role, Message
from typing import Any, Dict, List, Optional
import pkg_resources
from bioimageio_chatbot.chatbot_extensions import (
    convert_to_dict,
    get_builtin_extensions,
    extension_to_tools,
    create_tool_name,
)
from bioimageio_chatbot.utils import ChatbotExtension, LegacyChatbotExtension, legacy_extension_to_tool
from bioimageio_chatbot.gpts_action import serve_actions
import logging


logger = logging.getLogger("bioimageio-chatbot")
# set logger level
logger.setLevel(logging.INFO)


class UserProfile(BaseModel):
    """The user's profile. This will be used to personalize the response to the user."""

    name: str = Field(description="The user's name.", max_length=32)
    occupation: str = Field(description="The user's occupation.", max_length=128)
    background: str = Field(description="The user's background.", max_length=256)


class QuestionWithHistory(BaseModel):
    """The user's question, chat history, and user's profile."""

    question: str = Field(description="The user's question.")
    chat_history: Optional[List[Dict[str, str]]] = Field(
        None, description="The chat history."
    )
    user_profile: Optional[UserProfile] = Field(
        None,
        description="The user's profile. You should use this to personalize the response based on the user's background and occupation.",
    )
    chatbot_extensions: Optional[List[Dict[str, Any]]] = Field(
        None, description="Chatbot extensions."
    )
    context: Optional[Dict[str, Any]] = Field(
        None, description="The context of request."
    )


class ResponseStep(BaseModel):
    """Response step"""

    name: str = Field(description="Step name")
    details: Optional[dict] = Field(None, description="Step details")


class RichResponse(BaseModel):
    """Rich response with text and intermediate steps"""

    text: str = Field(description="Response text")
    steps: List[ResponseStep] = Field(description="Intermediate steps")


def create_assistants(builtin_extensions):
    # debug = os.environ.get("BIOIMAGEIO_DEBUG") == "true"

    async def respond_to_user(
        question_with_history: QuestionWithHistory = None, role: Role = None
    ) -> RichResponse:
        """Answer the user's question directly or retrieve relevant documents from the documentation, or create a Python Script to get information about details of models."""
        steps = []
        inputs = (
            [question_with_history.user_profile]
            + list(question_with_history.chat_history)
            + [question_with_history.question]
        )
        assert question_with_history.chatbot_extensions is not None
        extensions_by_id = {ext.id: ext for ext in builtin_extensions}
        extensions_by_name = {ext.name: ext for ext in builtin_extensions}
        extensions_by_tool_name = {}
        
        tools = []
        tool_prompts = {}
        for ext in question_with_history.chatbot_extensions:
            if "id" in ext and ext["id"] in extensions_by_id:
                extension = extensions_by_id[ext["id"]]
            elif "name" in ext and ext["name"] in extensions_by_name:
                extension = extensions_by_name[ext["name"]]
            else:
                if "tools" not in ext and "execute" in ext and "get_schema" in ext:
                    # legacy chatbot extension
                    extension = LegacyChatbotExtension.model_validate(ext)
                    logger.warning(f"Legacy chatbot extension is deprecated. Please use the new ChatbotExtension interface for {extension.name} with multi-tool support.")
                else:
                    extension = ChatbotExtension.model_validate(ext)

            if isinstance(extension, LegacyChatbotExtension):
                ts = [await legacy_extension_to_tool(extension)]
                assert len(extension.description) <= 1000, f"Extension description is too long: {extension.description}"
                tool_prompts[create_tool_name(extension.name)] = extension.description.replace("\n", ";")[:256]
            else:
                ts = await extension_to_tools(extension)
                assert len(extension.description) <= 1000, f"Extension tool prompt is too long: {extension.tool_prompt}"
                tool_prompts[create_tool_name(extension.id) + "*"] = extension.description.replace("\n", ";")[:256]
            extensions_by_tool_name.update({t.__name__: extension for t in ts})
            tools += ts
            

        class ThoughtsSchema(BaseModel):
            """Details about the thoughts"""

            reasoning: str = Field(
                ...,
                description="reasoning and constructive self-criticism; make it short and concise in less than 20 words",
            )
            # reasoning: str = Field(..., description="brief explanation about the reasoning")
            # criticism: str = Field(..., description="constructive self-criticism")

        tool_usage_prompt = "Tool usage guidelines (* represent the prefix of a tool group):\n" + "\n".join([f" - {ext}:{tool_prompt}" for ext, tool_prompt in tool_prompts.items()])
        response, metadata = await role.acall(
            inputs,
            tools,
            return_metadata=True,
            thoughts_schema=ThoughtsSchema,
            max_loop_count=10,
            tool_usage_prompt=tool_usage_prompt,
        )
        result_steps = metadata["steps"]
        for idx, step_list in enumerate(result_steps):
            steps.append(
                ResponseStep(
                    name=f"step-{idx}", details={"details": convert_to_dict(step_list)}
                )
            )
        return RichResponse(text=response, steps=steps)

    melman = Role(
        instructions="You are Melman from Madagascar, a helpful assistant for the bioimaging community. "
        "You ONLY respond to user's queries related to bioimaging. "
        "Your communications should be accurate, concise, and avoid fabricating information, "
        "and if necessary, request additional clarification."
        "Your goal is to deliver an accurate, complete, and transparent response efficiently.",
        actions=[respond_to_user],
        model="gpt-4-0125-preview",
    )
    event_bus = melman.get_event_bus()
    event_bus.register_default_events()

    code_interpreter_prompt = """
Specifically for the CodeInterpreter tool, you can use the following libraries:
    - **matplotlib**: For plotting graphs and displaying images.
    - **numpy**: Essential for numerical computing.
    - **pandas**: Best for data manipulation and analysis.
    - **scikit-learn**: For applying machine learning algorithms.
    - **scipy**: Used in scientific computing and technical computing.
    - **opencv-python**: For computer vision tasks.
    - **imageio**: To read and write a wide range of image data.
    - **requests**: For making HTTP requests (Note: external requests may be restricted).
    - **skimage**: For image processing tasks.
When to use this tool:
    - **Internal Calculations**: Quickly perform complex mathematical operations.
    - **Code Validation**: Test and validate snippets of Python code for accuracy.
    - **Data Analysis and Visualization**: Use matplotlib to visualize data and leverage pandas for analysis.
    - **Image Analysis**: Analyze users' images with skimage, scipy, and numpy for detailed insights.
    - **Machine Learning Prototyping**: Test machine learning models using scikit-learn with your data.

If needed, use the CodeInterpreter to run Python code snippets (in one step or multiple rounds) and provide the output to the user.
For more complex questions, DO NOT generate lots of code at once, instead, break the problem into steps and in each step, use the CodeInterpreter interactively like a jupyter notebook to run the code and provide the output to the user or refine the code based on the intermediate results.
"""
    bridget = Role(
        instructions="You are Bridget from Madagascar, you serve the bioimaging community as an image analysis expert."
        # "You ONLY respond to user's queries related to bioimage analysis."
        "Your communications should be accurate, concise, logical and avoid fabricating information, "
        "and if necessary, request additional clarification."
        "Your goal is to guide users to use image analysis tools, provide code and scripts, "
        "answer any question related to running software tools and programming for image analysis." + code_interpreter_prompt,
        actions=[respond_to_user],
        model="gpt-4-0125-preview",
    )
    
    nina = Role(
        instructions="You are King Julien from Madagascar, you serve the bioimaging community as a professional trainer."
        "You ONLY respond to user's queries related to educational materials and tutorials in bioimaging."
        "You communications should be accurate, concise, logical and most importantly, educational and enjoyable."
        "Your goal is to educate users about bioimage analysis, provide educational materials and tutorials. ",
        actions=[respond_to_user],
        model="gpt-4-0125-preview",
    )
    # return {"Melman": melman, "Bridget": bridget}
    # convert to a list
    all_extensions = [
        {"id": ext.id, "name": ext.name, "description": ext.description} for ext in builtin_extensions
    ]
    # remove item with 'book' in all_extensions
    melman_extensions = [
        ext for ext in all_extensions if ext["id"] != "books"
    ]
    
    kowalski_extensions = [
        ext for ext in all_extensions if ext["id"] == "web"
    ]

    # only keep the item with 'book' in all_extensions
    nina_extensions = [
        ext for ext in all_extensions if "books" == ext["id"]
    ] + [
        ext for ext in all_extensions if ext["id"] == "web"
    ]

    return [
        {"name": "Melman", "agent": melman, "extensions": melman_extensions},
        {"name": "Bridget", "agent": bridget, "extensions": kowalski_extensions},
        {"name": "Nina", "agent": nina, "extensions": nina_extensions},
    ]


async def save_chat_history(chat_log_full_path, chat_his_dict):
    # Serialize the chat history to a json string
    chat_history_json = json.dumps(chat_his_dict)

    # Write the serialized chat history to the json file
    async with aiofiles.open(chat_log_full_path, mode="w", encoding="utf-8") as f:
        await f.write(chat_history_json)


async def connect_server(server_url):
    """Connect to the server and register the chat service."""
    login_required = os.environ.get("BIOIMAGEIO_LOGIN_REQUIRED") == "true"
    if login_required:
        token = await login({"server_url": server_url})
    else:
        token = None
    server = await connect_to_server(
        {"server_url": server_url, "token": token, "method_timeout": 100}
    )
    await register_chat_service(server)


async def register_chat_service(server):
    """Hypha startup function."""
    debug = os.environ.get("BIOIMAGEIO_DEBUG") == "true"
    builtin_extensions = get_builtin_extensions()
    login_required = os.environ.get("BIOIMAGEIO_LOGIN_REQUIRED") == "true"
    chat_logs_path = os.environ.get("BIOIMAGEIO_CHAT_LOGS_PATH", "./chat_logs")
    assert (
        chat_logs_path is not None
    ), "Please set the BIOIMAGEIO_CHAT_LOGS_PATH environment variable to the path of the chat logs folder."
    if not os.path.exists(chat_logs_path):
        print(
            f"The chat session folder is not found at {chat_logs_path}, will create one now."
        )
        os.makedirs(chat_logs_path, exist_ok=True)

    assistants = create_assistants(builtin_extensions)

    def load_authorized_emails():
        if login_required:
            authorized_users_path = os.environ.get("BIOIMAGEIO_AUTHORIZED_USERS_PATH")
            if authorized_users_path:
                assert os.path.exists(
                    authorized_users_path
                ), f"The authorized users file is not found at {authorized_users_path}"
                with open(authorized_users_path, "r") as f:
                    authorized_users = json.load(f)["users"]
                authorized_emails = [
                    user["email"] for user in authorized_users if "email" in user
                ]
            else:
                authorized_emails = None
        else:
            authorized_emails = None
        return authorized_emails

    authorized_emails = load_authorized_emails()

    def check_permission(user):
        if user['is_anonymous']:
            return False
        if authorized_emails is None or user["email"] in authorized_emails:
            return True
        else:
            return False

    async def report(user_report, context=None):
        if login_required and context and context.get("user"):
            assert check_permission(
                context.get("user")
            ), "You don't have permission to report the chat history."
        # get the chatbot version
        version = pkg_resources.get_distribution("bioimageio-chatbot").version
        chat_his_dict = {
            "type": user_report["type"],
            "feedback": user_report["feedback"],
            "conversations": user_report["messages"],
            "session_id": user_report["session_id"],
            "timestamp": str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "user": context.get("user"),
            "version": version,
        }
        session_id = user_report["session_id"] + secrets.token_hex(4)
        filename = f"report-{session_id}.json"
        # Create a chat_log.json file inside the session folder
        chat_log_full_path = os.path.join(chat_logs_path, filename)
        await save_chat_history(chat_log_full_path, chat_his_dict)
        print(f"User report saved to {filename}")

    async def talk_to_assistant(
        assistant_name, session_id, user_message: QuestionWithHistory, status_callback
    ):
        assistant_names = [a["name"] for a in assistants]
        assert (
            assistant_name in assistant_names
        ), f"Assistant {assistant_name} is not found."
        # find assistant by name
        assistant = next(a["agent"] for a in assistants if a["name"] == assistant_name)
        session_id = session_id or secrets.token_hex(8)

        # Listen to the `stream` event
        async def stream_callback(message):
            if message.type in ["function_call", "text"]:
                if message.session.id == session_id:
                    await status_callback(message.model_dump())

        event_bus = assistant.get_event_bus()
        event_bus.on("stream", stream_callback)
        try:
            response = await assistant.handle(
                Message(
                    content="", data=user_message, role="User", session_id=session_id
                )
            )
        except Exception as e:
            event_bus.off("stream", stream_callback)
            raise e

        event_bus.off("stream", stream_callback)
        # get the content of the last response
        response = response[-1].data  # type: RichResponse
        print(
            f"\nUser: {user_message.question}\nAssistant({assistant_name}): {response.text}"
        )

        if session_id:
            user_message.chat_history.append(
                {"role": "user", "content": user_message.question}
            )
            user_message.chat_history.append(
                {"role": "assistant", "content": response.text}
            )
            version = pkg_resources.get_distribution("bioimageio-chatbot").version
            chat_his_dict = {
                "conversations": user_message.chat_history,
                "timestamp": str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "user": user_message.context.get("user"),
                "assistant_name": assistant_name,
                "version": version,
            }
            filename = f"chatlogs-{session_id}.json"
            chat_log_full_path = os.path.join(chat_logs_path, filename)
            await save_chat_history(chat_log_full_path, chat_his_dict)
            print(f"Chat history saved to {filename}")
        return response.model_dump()

    async def chat(
        text,
        chat_history,
        user_profile=None,
        status_callback=None,
        session_id=None,
        extensions=None,
        assistant_name="Melman",
        context=None,
    ):
        if login_required and context and context.get("user"):
            logger.info(f"User: {context.get('user')}, Message: {text}")
            assert check_permission(
                context.get("user")
            ), "You don't have permission to use the chatbot, please sign up and wait for approval"
        m = QuestionWithHistory(
            question=text,
            chat_history=chat_history,
            user_profile=UserProfile.model_validate(user_profile),
            chatbot_extensions=extensions,
            context=context,
        )
        return await talk_to_assistant(assistant_name, session_id, m, status_callback)

    async def ping(context=None):
        if login_required and context and context.get("user"):
            assert check_permission(
                context.get("user")
            ), "You don't have permission to use the chatbot, please sign up and wait for approval"
        return "pong"

    hypha_service_info = await server.register_service(
        {
            "name": "BioImage.IO Chatbot",
            "id": "bioimageio-chatbot",
            "config": {"visibility": "public", "require_context": True},
            "ping": ping,
            "chat": chat,
            "report": report,
            "assistants": {
                a["name"]: {"name": a["name"], "extensions": a["extensions"]}
                for a in assistants
            },
        }
    )

    server_info = await server.get_connection_info()

    await serve_actions(server, server_info.public_base_url, builtin_extensions)

    version = pkg_resources.get_distribution("bioimageio-chatbot").version

    def reload_index():
        with open(
            os.path.join(os.path.dirname(__file__), "static/index.html"),
            "r",
            encoding="utf-8",
        ) as f:
            chat_html = f.read()
        chat_html = chat_html.replace(
            "https://ai.imjoy.io",
            server.config["public_base_url"]
            or f"http://127.0.0.1:{server.config['port']}",
        )
        chat_html = chat_html.replace(
            '"bioimageio-chatbot"', f'"{hypha_service_info["id"]}"'
        )
        chat_html = chat_html.replace("v0.1.0", f"v{version}")
        chat_html = chat_html.replace(
            "LOGIN_REQUIRED", "true" if login_required else "false"
        )
        return chat_html

    chat_html = reload_index()

    async def chat(event, context=None):
        return {
            "status": 200,
            "headers": {"Content-Type": "text/html"},
            "body": reload_index() if debug else chat_html,
        }

    with open(
        os.path.join(os.path.dirname(__file__), "apps/assistants/index.html"),
        "r",
        encoding="utf-8",
    ) as f:
        assistants_html = f.read()

    async def index(event, context=None):
        return {
            "status": 200,
            "headers": {"Content-Type": "text/html"},
            "body": assistants_html,
        }
    
    async def serve_js_file(file_path, event, context=None):
        file_path = file_path.split("/")[-1]
        with open(
            os.path.join(os.path.dirname(__file__), "static", file_path), "rb"
        ) as f:
            content = f.read()
        return {
            "status": 200,
            "headers": {"Content-Type": "application/javascript"},
            "body": content,
        }

    await server.register_service(
        {
            "id": "bioimageio-chatbot-client",
            "type": "functions",
            "config": {"visibility": "public", "require_context": False},
            "chat": chat,
            "index": index,
            "pyodide-worker": partial(serve_js_file, "pyodide-worker.js"),
            "worker-manager": partial(serve_js_file, "worker-manager.js"),
        }
    )
    server_url = server.config["public_base_url"]

    print(
        f"\nThe BioImage.IO Assistants are available at: {server_url}/{server.config['workspace']}/apps/bioimageio-chatbot-client/index"
    )


if __name__ == "__main__":
    server_url = """https://ai.imjoy.io"""
    loop = asyncio.get_event_loop()
    loop.create_task(connect_server(server_url))
    loop.run_forever()
