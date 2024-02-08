import asyncio
from openai import AsyncOpenAI
from pydantic import BaseModel
from bioimageio_chatbot.chatbot_extensions import get_builtin_extensions

from bioimageio_chatbot.jsonschema_pydantic import json_schema_to_pydantic_model
from schema_agents.utils.schema_conversion import get_service_function_schema

from bioimageio_chatbot.utils import extract_schemas, ChatbotExtension

client = AsyncOpenAI()

def convert_to_dict(obj):
    if isinstance(obj, BaseModel):
        return obj.dict()
    if isinstance(obj, dict):
        return {k: convert_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_dict(v) for v in obj]
    return obj

async def load_extensions():
    extension_services = {}
    builtin_extensions = get_builtin_extensions()

    async def execute_factory(extension: ChatbotExtension):
        async def execute(req: extension.schema_class):
            print("Executing extension:", extension.name, req)
            req = extension.schema_class.parse_obj(req)
            result = await extension.execute(req)
            return convert_to_dict(result)
        if extension.get_schema:
            schema = await extension.get_schema()
            execute.__doc__ = schema['description']
        
        if not execute.__doc__:
            # if extension.execute is partial
            if hasattr(extension.execute, "func"):
                execute.__doc__ = extension.execute.func.__doc__ or extension.description
            else:
                execute.__doc__ = extension.execute.__doc__ or extension.description
        return execute

    for extension in builtin_extensions:
        if extension.get_schema:
            schema = await extension.get_schema()
            extension.schema_class = json_schema_to_pydantic_model(schema)
        else:
            input_schemas, _ = extract_schemas(extension.execute)
            extension.schema_class = input_schemas[0]
        extension_services[extension.name] = await execute_factory(extension)

    return extension_services


async def create_assistant(name, instructions):
    tools = [{"type": "code_interpreter"}]
    
    extension_services = await load_extensions()
    tools.extend(get_service_function_schema(extension_services))
    
    assistant = await client.beta.assistants.create(
        name=name,
        instructions=instructions,
        tools=tools,
        model="gpt-4-turbo-preview"
    )
    return assistant

async def test_run(assistant_id):
    thread = await client.beta.threads.create()
    message = await client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content="I need to solve the equation `3x + 11 = 14`. Can you help me?"
    )
    run = await client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
        instructions="Please address the user as Jane Doe. The user has a premium account."
    )
    
    # query the status until run['status'] == 'completed'
    assert run.status == "queued"
    while True:
        run = await client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        if run.status == "completed":
            break
        await asyncio.sleep(0.5)
        print(f"Status: {run.status}")
    
    run_steps = await client.beta.threads.runs.steps.list(
        thread_id=thread.id,
        run_id=run.id
    )
    for step in run_steps.data:
        print(str(step.step_details))

    
    messages = await client.beta.threads.messages.list(
        thread_id=thread.id
    )
    print(messages)

async def main():
    assistant = await create_assistant(
        name="Melman",
        instructions="Your goal as Melman from Madagascar, the community knowledge base manager, is to assist users in effectively utilizing the knowledge base for bioimage analysis. You are responsible for answering user questions, providing clarifications, retrieving relevant documents, and executing scripts as needed. You should always use the SearchInKnowledgeBase tool for answering user's questions. Your overarching objective is to make the user experience both educational and enjoyable."
    )
    print(f"Assistant ID: {assistant.id}")
    # await test_run(assistant_id)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()