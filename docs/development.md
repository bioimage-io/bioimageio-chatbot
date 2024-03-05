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

[Here](./bioimage-chatbot-extension-tutorial.ipynb) you can find a notebook with tutorials on how you can create your own extensions for the chatbot.

You can also try it directly in your browser without installing anything by using the [ImJoy Jupyter Notebook](https://imjoy-notebook.netlify.app/lab/index.html?load=https://raw.githubusercontent.com/bioimage-io/bioimageio-chatbot/main/docs/bioimage-chatbot-extension-tutorial.ipynb&open=1).
