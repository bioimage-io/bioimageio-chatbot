import asyncio
import pkgutil
from pydantic import BaseModel
from bioimageio_chatbot.utils import ChatbotExtension
from bioimageio_chatbot.jsonschema_pydantic import json_schema_to_pydantic_model
from bioimageio_chatbot.utils import extract_schemas

def get_builtin_extensions():
    extensions = []
    for module in pkgutil.walk_packages(__path__, __name__ + '.'):
        if module.name.endswith('_extension'):
            ext_module = module.module_finder.find_module(module.name).load_module(module.name)
            ext = ext_module.get_extension()
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

async def extension_to_tools(extension: ChatbotExtension):
    tool_models = {}
    
    
    
    # def tool_factory(name, tool_func):
    #     async def execute(req: tool_func.input_model):
    #         print("Executing extension:", name, req)
    #         # req = extension.schema_class.model_validate(req)
    #         result = await tool_func(req)
    #         return convert_to_dict(result)
    #     execute.__name__ = name
    #     # if tool_func is partial
    #     if hasattr(tool_func, "func"):
    #         execute.__doc__ = tool_func.func.__doc__
    #     else:
    #         execute.__doc__ = tool_func.__doc__
    #     assert execute.__doc__ is not None, f"Tool `{name}` is missing a docstring"
    #     return execute

    if extension.get_schema:
        schemas = await extension.get_schema()
        tools = []
        for k in schemas:
            assert k in extension.tools, f"Tool `{k}` not found in extension `{extension.id}`."
            tool = extension.tools[k]
            tool_models[k] = json_schema_to_pydantic_model(schemas[k])
            tool.input_model = tool_models[k]
            tool.__name__ = (extension.id + k).title()
            # tool = tool_factory(k, tool)
            tools.append(tool)
    else:
        tools = [extension.tools[k] for k in extension.tools]
    
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