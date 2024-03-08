# Developing Chatbot Extensions

## Introduction
The BioImage.IO Chatbot offers a framework designed for easy extensibility, allowing developers to enrich its capabilities with custom extensions. This guide walks you through the process of developing and integrating new extensions into the chatbot, emphasizing the minimal requirements and the steps involved in using ImJoy to interact with the chatbot.

Extensions must expose a callable function that adheres to a specific interface: it should accept a dictionary of parameters as its single argument and return a dictionary containing the results of its operations. This design facilitates seamless integration and communication between the chatbot and its extensions.

You can extend the chatbot using [ImJoy](https://imjoy.io).
Below are examples demonstrating how to register an extension with the chatbot using both JavaScript and Python in ImJoy:

You can try them here: https://imjoy.io/lite?plugin=https://if.imjoy.io

### JavaScript Example
```javascript
const chatbot = await api.createWindow({
    src: "https://chat.bioimage.io/public/apps/bioimageio-chatbot-client/chat",
    name:"BioImage.IO Chatbot",
});
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
        };
    },
    tools: {
        my_tool(config) {
            console.log(config.my_param);
            return {result: "success"};
        }
    }
});
```

### Python Example
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
    };

def my_tool(config):
    print(config["my_param"]);
    return {"result": "success"};

chatbot = await api.createWindow(
    src="https://chat.bioimage.io/public/apps/bioimageio-chatbot-client/chat",
    name="BioImage.IO Chatbot",
);
await chatbot.registerExtension({
    "id": "my-extension",
    "name": "My Extension",
    "description": "This is my extension",
    "get_schema": get_schema,
    "tools": {
        "my_tool": my_tool
    }
});
```

## Tutorial
For an in-depth understanding, refer to [our detailed tutorial](./bioimage-chatbot-extension-tutorial.ipynb), accessible directly through the ImJoy Jupyter Notebook in your browser without installation. [Click here to launch the notebook](https://imjoy-notebook.netlify.app/lab/index.html?load=https://raw.githubusercontent.com/bioimage-io/bioimageio-chatbot/main/docs/bioimage-chatbot-extension-tutorial.ipynb&open=1).

## Extension Development Details

### `tools` and `get_schema`
When developing extensions, it's essential to define the `tools` and `get_schema` functionalities carefully:
- **`tools`**: Represents the set of functions your extension offers, each accepting configuration parameters as input. These functions should carry out specific tasks and return their results in a dictionary.
- **`get_schema`**: Returns a JSON schema for each tool function, specifying the structure and types of the expected parameters. This schema is crucial for instructing the chatbot to generate the correct input paramters and validate the inputs and ensuring they adhere to the expected format. Importantly, the chatbot uses the title and description for each field to understand what expected for the tool will generating a function call to run the tool (also see the detailed instructions below).

### Notes on Function Input/Output
The input and output of tool functions are restricted to primitive types (e.g., numbers, strings) that can be encoded in JSON. This limitation ensures compatibility and facilitates the exchange of data between the chatbot and extensions.

### Importance of Detailed Descriptions
Providing a detailed description for your extension and its arguments is vital. These descriptions assist the chatbot in correctly invoking the tools and help the chatbot understand the functionality and purpose of your extension. Ensure that each argument is accompanied by a clear title and a comprehensive description to improve usability and interaction quality of the chatbot.

By adhering to these guidelines, you will enhance the clarity, utility, and ease of integration of your chatbot extensions, contributing to a richer ecosystem of tools within the BioImage.IO community.
