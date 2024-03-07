# Developing Chatbot Extensions

## Introduction
BioImage.IO Chatbot is designed to be easily extensible. This document describes how to develop and integrate new extensions into the chatbot.

The minimal requirement for an extension is to have a function that can be called from the chatbot. The function should take a single argument, which is a dictionary of parameters. The function should return a dictionary with the result of the operation.

You can use ImJoy to interact with the chatbot. After creating the chatbot window in ImJoy, the chatbot extension can be registered with the chatbot using the `registerExtension` method. For example in javascript:
```javascript
const chatbot = await api.createWindow({
    src: "https://chat.bioimage.io/public/apps/bioimageio-chatbot-client/chat",
    name:"BioImage.IO Chatbot",
})
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
        my_tool(config) {
            console.log(config.my_param)
            return {result: "success"}
        }
    }
})
```

Or in Python:
```python
from imjoy_rpc import api

def get_schema():
    return {
        "my_tool": {
            "type": "object",
            "properties": {
                "my_param": {
                    "type": "number",
                    "description": "This is my parameter"
                }
            }
        }
    }

def my_tool(config):
    print(config["my_param"])
    return {"result": "success"}

chatbot = await api.createWindow(
    src="https://chat.bioimage.io/public/apps/bioimageio-chatbot-client/chat",
    name="BioImage.IO Chatbot",
)
await chatbot.registerExtension({
    "id": "my-extension",
    "name": "My Extension",
    "description": "This is my extension",
    "get_schema": get_schema,
    "tools": {
        "my_tool": my_tool
    }
})

```

## Tutorial

For a more compelte example, we provide [a notebook here](./bioimage-chatbot-extension-tutorial.ipynb). You can also try it directly in your browser without installing anything by using the ImJoy Jupyter Notebook, [click here to launch the notebook](https://imjoy-notebook.netlify.app/lab/index.html?load=https://raw.githubusercontent.com/bioimage-io/bioimageio-chatbot/main/docs/bioimage-chatbot-extension-tutorial.ipynb&open=1).
