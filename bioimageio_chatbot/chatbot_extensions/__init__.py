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
            exts = ext_module.get_extensions()
            for ext in exts:
                if ext.name in [e.name for e in extensions]:
                    raise ValueError(f"Extension name {ext.name} already exists.")
            extensions.extend(exts)
            
    return extensions

def convert_to_dict(obj):
    if isinstance(obj, BaseModel):
        return obj.dict()
    if isinstance(obj, dict):
        return {k: convert_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_dict(v) for v in obj]
    return obj

async def extension_to_tool(extension: ChatbotExtension):
    if extension.get_schema:
        schema = await extension.get_schema()
        extension.schema_class = json_schema_to_pydantic_model(schema)
    else:
        input_schemas, _ = extract_schemas(extension.execute)
        extension.schema_class = input_schemas[0]

    assert extension.schema_class, f"Extension {extension.name} has no valid schema class."

    # NOTE: Right now, the first arguments has to be req
    async def execute(req: extension.schema_class):
        print("Executing extension:", extension.name, req)
        # req = extension.schema_class.parse_obj(req)
        result = await extension.execute(req)
        return convert_to_dict(result)

    execute.__name__ = extension.name

    if extension.get_schema:
        execute.__doc__ = schema['description']
    
    if not execute.__doc__:
        # if extension.execute is partial
        if hasattr(extension.execute, "func"):
            execute.__doc__ = extension.execute.func.__doc__ or extension.description
        else:
            execute.__doc__ = extension.execute.__doc__ or extension.description
    return execute
    
if __name__ == "__main__":
    print(get_builtin_extensions())