import asyncio
import re
import pkgutil
from pydantic import BaseModel
from bioimageio_chatbot.utils import ChatbotExtension
from bioimageio_chatbot.jsonschema_pydantic import json_schema_to_pydantic_model
from schema_agents import schema_tool

def get_builtin_extensions():
    extensions = []
    for module in pkgutil.walk_packages(__path__, __name__ + '.'):
        if module.name.endswith('_extension'):
            ext_module = module.module_finder.find_module(module.name).load_module(module.name)
            exts = ext_module.get_extension()
            if isinstance(exts, ChatbotExtension):
                exts = [exts]
            for ext in exts:
                if not isinstance(ext, ChatbotExtension):
                    print(f"Failed to load chatbot extension: {module.name}.")
                    continue
                if ext.id in [e.id for e in extensions]:
                    raise ValueError(f"Extension name {ext.id} already exists.")
                extensions.append(ext)
            
    return extensions

def convert_to_dict(obj):
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: convert_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_dict(v) for v in obj]
    return obj

def create_tool_name(ext_id, tool_id=""):
    text = f"{ext_id}_{tool_id}"
    text = text.replace("-", " ").replace("_", " ").replace(".", " ")
    words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+', text)
    return ''.join(word if word.istitle() else word.capitalize() for word in words)

def tool_factory(ext_id, tool_id, ext_tool, schema):
    input_model = json_schema_to_pydantic_model(schema)
    ext_tool.__name__ = create_tool_name(ext_id, tool_id)
    ext_tool.__doc__ = input_model.__doc__
    return schema_tool(ext_tool, input_model=input_model)

async def extension_to_tools(extension: ChatbotExtension):

    if extension.get_schema:
        schemas = await extension.get_schema()
        tools = []
        for k in schemas:
            assert k in extension.tools, f"Tool `{k}` not found in extension `{extension.id}`."
            ext_tool = extension.tools[k]
            tool = tool_factory(extension.id, k, ext_tool, schemas[k])
            tools.append(tool)
    else:
        tools = []
        for k in extension.tools:
            ext_tool = extension.tools[k]
            ext_tool.__name__ = create_tool_name(extension.id, k)
            tools.append(ext_tool)
    
    return tools

async def main():
    extensions = get_builtin_extensions()
    tools = []
    for svc in extensions:
        tool = await extension_to_tools(svc)
        tools.append(tool)
    print(tools)

if __name__ == "__main__":
    asyncio.run(main())