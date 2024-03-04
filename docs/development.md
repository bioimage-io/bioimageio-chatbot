# Developing Chatbot Extensions

## Introduction
BioImage.IO Chatbot is designed to be easily extensible. This document describes how to develop and integrate new extensions into the chatbot.

The minimal requirement for an extension is to have a function that can be called from the chatbot. The function should take a single argument, which is a dictionary of parameters. The function should return a dictionary with the result of the operation.

After creating the chatbot window, the extension can be registered with the chatbot using the `registerExtension` method. For example:
```javascript

function my_tool_function(kwargs) {
    console.log(kwargs)
    return {result: "success"}
}

chatbot.registerExtension({
    id: "my-extension",
    name: "My Extension",
    description: "This is my extension",
    get_schema() {
        return {
            my_tool: {
                type: "object",
                properties: {
                    my_param: {
                        type: "number",
                        description: "This is my parameter"
                    }
                }
            }
        }
    },
    tools: {
        my_tool: my_tool_function
    }
})
```

Here is an example of a simple extension that can be used to control a microscope.

```python
from pydantic import BaseModel
from imjoy_rpc import api
    
class MoveStageInput(BaseModel):
    x: float = Field(..., description="x offset")
    y: float = Field(..., description="y offset")

class SnapImageInput(BaseModel):
    exposure: float = Field(..., description="exposure time")

def move_stage(kwargs):
    config = MoveStageInput(**kwargs)
    print(config.x, config.y)

    return "success"

def snap_image(kwargs):
    config = SnapImageInput(**kwargs)
    print(config.exposure)

    return "success"

async def setup():
    chatbot = await api.createWindow(src="https://chat.bioimage.io/public/apps/bioimageio-chatbot-client/chat")
    
    def get_schema():
        return {
            "move_stage": MoveStageInput.schema(),
            "snap_image": SnapImageInput.schema()
        }

    extension = {
        "_rintf": True,
        "id": "squid-control",
        "name": "Squid Microscope Control",
        "description": "Contorl the squid microscope....",
        "get_schema": get_schema,
        "tools": {
            "move_stage": move_stage,
            "snap_image": snap_image,
        }
    }
    await chatbot.registerExtension(extension)

api.export({"setup": setup})
```