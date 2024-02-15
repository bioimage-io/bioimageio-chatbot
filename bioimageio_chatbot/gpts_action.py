import asyncio
from openai import AsyncOpenAI
from pydantic import BaseModel
from bioimageio_chatbot.chatbot_extensions import get_builtin_extensions, extension_to_tool
from schema_agents.utils.schema_conversion import get_service_openapi_schema
from imjoy_rpc.hypha import login, connect_to_server

client = AsyncOpenAI()

async def load_extensions():
    extension_services = {}
    builtin_extensions = get_builtin_extensions()
    for extension in builtin_extensions:
        extension_services[extension.name] = await extension_to_tool(extension)
    return extension_services

async def serve_actions(server, server_url):
    extension_services = await load_extensions()
    svc = {
        "id": "bioimageio-chatbot-extensions-api",
        "name": "BioImage.io Chatbot Extensions",
        "description": "A collection of chatbot extensions for facilitate user interactions with external documentation, services and tools.",
        "config": {
            "visibility": "public",
            "require_context": False
        },
    }
    svc.update(extension_services)
    workspace = server.config['workspace']
    service_id = "bioimageio-chatbot-extensions-api"
    openapi_schema = get_service_openapi_schema(svc, f"{server_url}/{workspace}/services/{service_id}")
    svc["get_openapi_schema"] = lambda : openapi_schema

    service_info = await server.register_service(svc)
    print(f"Service registered, openapi schema: {server_url}/services/call?service_id={service_info['id']}&function_key=get_openapi_schema")

async def start_server(server_url):
    token = await login({"server_url": server_url})
    server = await connect_to_server({"server_url": server_url, "token": token, "method_timeout": 100})
    print(f"Connected to server: {server_url}")
    await serve_actions(server, server_url)


if __name__ == "__main__":
    server_url = "https://staging.chat.bioimage.io/"
    loop = asyncio.get_event_loop()
    loop.create_task(start_server(server_url))
    loop.run_forever()