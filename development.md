# 🤖 BioImage.IO Chatbot Development Guide 💻

## Contents
- [Technical Overview](#technical-overview)
- [Setup your Chatbot Locally](#setup-your-chatbot-locally)
- [Command-line Interface](#command-line-interface)
- [Running the BioImage.IO Chatbot in a Docker Container](#running-the-bioimageio-chatbot-in-a-docker-container)
- [Contribution Process](#contribution-process)
- [Developing Chatbot Extensions](#developing-chatbot-extensions)


## Technical Overview
### Chatbot Interface

After following the installation guidelines from the [README](/README.md), the chat interface will resemble Figure 1.

![BioImage.IO-Chatbot](./docs/screenshots/chat-interface.png)
*Figure 1. The chat interface of the BioImage.IO Chatbot.*

Users can input their profiles as depicted in Figure 2. 
![user-profile](./docs/screenshots/user-profile.png)
*Figure 2. Users can personalize responses by clicking `Edit Profile` and save their settings for future conversations by clicking `Save`.*

As of today, our chatbot integrates 6 extensions including document search in bioimage.io knowledge base, tools search on Bioimage Informatics Index (biii.eu), bioimage topics search in Bioimage Archive and Image.cs Forum, web search, and information search in Bioimage Model Zoo. The document search utilizes knowledge bases from the following pivotal communities: bioimage.io [2], Imjoy [3], deepimageJ [4], ImageJ [5], bio.tools [6], and scikit-image [7]. We also allow users to specify a preferred extension for information retrieval, as shown in Figure 3. If an extension is designated, the chatbot sources information using the specific extension and its corresponding source. Otherwise, it uses an intelligent selection process driven by a schema-based agent to choose the most relevant extension based on the user's query. 

![channels](./docs/screenshots/extensions.png)
*Figure 3. Users can personalize the conversation by selecting a specific channel from the ‘Knowledge Base Channel’.*

#### Building the Knowledge Base

The knowledge base is efficiently and collaboratively constructed by downloading documentation from given URLs. These can be repositories, PDFs, or other forms of documentation. We use a regular expression splitter to segment the documentation into manageable chunks for efficient and accurate retrieval. These chunks are then embedded and stored as vectors in a FAISS [1]-based vector database.

### Schema-Based Agent Design

The chatbot's ability to understand and respond to user queries is substantially improved by employing a schema-based agent design. Unlike traditional context-based models, our approach utilizes predefined schemas to guide the conversation and information retrieval process. 

The schema-based agent operates on the function-call LLM [8], and uses input and output schemas to generate text output. Within this implementation, we construct a customer service chatbot by defining a role class, as shown in Figure 4.

![role_create](./docs/screenshots/role_create.png)
*Figure 4. Creation of a chatbot role class named ‘CustomerServiceRole’ by defining fields of the role class.*

### Extensions
The BioImage.IO Chatbot employs diverse methods to generate responses, currently encompassing five distinct response modes. The response mode is chosen by the schema-based agent based on the user's query and the selected channel.

#### Search BioImage Docs
This extension allows the chatbot to search information in a community-driven bioimage related knowledge base. With a specific query, the chatbot extracts essential elements from the user's question to fetch information from the relevant documentation. 

    ![direct-response](./docs/screenshots/search-bioimage-docs.png)
    *Figure 6. Search in Bioimage Knolwedge base documentation.*

#### Search BioImage Information Index (biii.eu)
This extension allows the chatbot to search online software tool in biii.eu.
    ![search-biii](./docs/screenshots/search-biii.png)
    *Figure 7. Search in biii.eu.*

The process begins with an initial response based on the user's query (`request`), which serves as a foundation for generating a new `query` for targeted information retrieval. This is combined with user profile data (`user_info`) and the query to produce a comprehensive final response.

#### Search Bioimage Archive
This extension allows the chatbot to search for dataset index in bioimage archive. 
    ![search-bioimage-archive](./docs/screenshots/search-bioimage-archive.png)
    *Figure 8. Search in bioimage archive.*

#### Search image.sc Forum
This extension allows the chatbot to search bioimage related topics and software issues in the image.sc forum.
    ![search-image-sc](./docs/screenshots/search-image-forum.png)
    *Figure 9. Search in image.sc forum.*

#### Search Web
This extension allows the chatbot to search for information from the web. This extension is triggered while the chatbot realizes it can not find relevant information from the knowledge base.

    ![search-web](./docs/screenshots/search-web.png)
    *Figure 10. Search in the web.*


#### BioImage Model Zoo
This mode is designed for queries requiring detailed model information or specific actions, generating and executing Python scripts for tailored solutions.

    ![script-gen-exe-retrieval](./docs/screenshots/search-model-zoo.png)
    *Figure 11. Scripting retrieval for complex queries.*

It involves creating a `ModelZooInfoScript` schema with fields like `request`, `user info`, and `script`, where `script` is Python code for API interactions or data manipulation. The final response is formulated by integrating the script's output with the `request` and `user info`.

### References

1. [FAISS](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)
2. [Bioimage.io](https://bioimage.io/docs/#/)
3. [Imjoy](https://imjoy.io/docs/#/)
4. [DeepImageJ](https://deepimagej.github.io/)
5. [ImageJ](https://imagej.net)
6. [bio.tools](https://bio.tools)
7. [scikit-image](https://scikit-image.org/docs/stable/)
8. [Function-Calling API](https://openai.com/blog/function-calling-and-other-api-updates)
9. [CellPose](https://www.cellpose.org)

## Setup your Chatbot Locally

If you want to run the chatbot server locally, you need to have an OpenAI API key. You can get one by signing up at [OpenAI](https://beta.openai.com/). Once you have your API key, you can install the chatbot package via pip and set the environment variables:

```bash
pip install bioimageio-chatbot
```

```bash
export OPENAI_API_KEY=sk-xxxxxxxx # Required
export BIOIMAGEIO_KNOWLEDGE_BASE_PATH=/path/to/bioimageio-knowledge-base  # Optional, default to ./bioimageio-knowledge-base 
export BIOIMAGEIO_CHAT_LOGS_PATH=./chat-logs # Optional, default to ./chat-logs
```

The chatbot server backend has been tested on Ubuntu and MacOS, it should work on Windows as well.

## Command-line Interface

BioImage.IO Chatbot comes with a command-line interface to facilitate server management, connection to external servers, and knowledge base creation.

You can access the command-line interface by running `python -m bioimageio_chatbot` or the `bioimageio-chatbot` command.

Below are the available commands and options:

### Initialize Knowledge Base

To initialize the knowledge base, use the `init` command:

```bash
python -m bioimageio_chatbot init
```

This will load the knowledge base from the location specified by the `BIOIMAGEIO_KNOWLEDGE_BASE_PATH` environment variable, or use the default path `./bioimageio-knowledge-base`. If the knowledge base is not found, it will be downloaded from the predefined URL (by default, it uses https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimageio-knowledge-base. It can be overridden with `BIOIMAGEIO_KNOWLEDGE_BASE_URL`).

NOTE: It may take some time to download the knowledge base depending on your internet connection. 
**Example:**

```bash
export BIOIMAGEIO_KNOWLEDGE_BASE_PATH="./my_knowledge_base"
python -m bioimageio_chatbot init
```

After running the `init` command, it will list the databases loaded into the knowledge base.

#### Start Server

To start your own server entirely, use the `start-server` command:

```bash
python -m bioimageio_chatbot start-server [--host HOST] [--port PORT] [--public-base-url PUBLIC_BASE_URL]
```

**Options:**

- `--host`: The host address to run the server on (default: `0.0.0.0`)
- `--port`: The port number to run the server on (default: `9000`)
- `--public-base-url`: The public base URL of the server (default: `http://127.0.0.1:9000`)
- `--login-required`: Whether to require users to log in before accessing the chatbot (default to not require login)

**Example:**

```bash
export OPENAI_API_KEY=sk-xxxxxxxx
export BIOIMAGEIO_KNOWLEDGE_BASE_PATH=./bioimageio-knowledge-base
export BIOIMAGEIO_CHAT_LOGS_PATH=./chat-logs
python -m bioimageio_chatbot start-server --host=0.0.0.0 --port=9000
```
This will create a local server, and the BioImage.IO Chatbot is available at: https://bioimage.io/chat?server=http://127.0.0.1:9000

Open the link in a browser, and you will see the chat interface.

Please note that the chatbot server may not be accessible to users outside your local network.

A user guide and technical overview can be found [here](./docs/technical-overview.md).

To be able to share your chatbot service over the internet (especially for users outside your local network), you will need to expose your server publicly. Please, see [Connect to Server](#connect-to-server)


#### Connect to Server

To help you share your chatbot with users external to your local network, you can use our public [BioEngine](https://aicell.io/project/bioengine/) server as a proxy.

To connect to an external BioEngine server, use the `connect-server` command:

```bash
python -m bioimageio_chatbot connect-server [--server-url SERVER_URL]
```

**Options:**

- `--server-url`: The URL of the external BioEngine server to connect to (default: `https://ai.imjoy.io`)
- `--login-required`: Whether to require users to log in before accessing the chatbot (default to not require login)

**Example:**

```bash
export OPENAI_API_KEY=sk-xxxxxxxx
export BIOIMAGEIO_KNOWLEDGE_BASE_PATH=./bioimageio-knowledge-base
export BIOIMAGEIO_CHAT_LOGS_PATH=./chat-logs
python -m bioimageio_chatbot connect-server --server-url=https://ai.imjoy.io
```

First, you will be asked to log in with a hypha account. Either your GitHub or Google account can be reused. Then, the following message containing a link to the chatbot will be displayed: 'The BioImage.IO Chatbot is available at: https://bioimage.io/chat?server=https://ai.imjoy.io'

Leave your chatbot running to enable users inside or outside your network to access it from this URL.

#### User Management

If you set `--login-required` when running `start-server` or `connect-server`, users will be required to log in before accessing the chatbot. The chatbot will then collect the user's GitHub or Google account information and store it its logs for future analysis.

You can also provide an optional environment variable `BIOIMAGEIO_AUTHORIZED_USERS_PATH` for the chatbot to load a list of authorized users. The file should be a JSON file containing a list of GitHub or Google account names. For example:

```json
{
    "users": [
        {"email": "user1@email.org"}
    ]
}
```

#### Create Knowledge Base

To create a new knowledge base, use the `create-knowledge-base` command:

```bash
python -m bioimageio_chatbot create-knowledge-base [--output-dir OUTPUT_DIR]
```

**Options:**

- `--output-dir`: The directory where the knowledge base will be created (default: `./bioimageio-knowledge-base`)

**Example:**

```bash
export OPENAI_API_KEY=sk-xxxxxxxx
export BIOIMAGEIO_KNOWLEDGE_BASE_PATH=./bioimageio-knowledge-base
python -m bioimageio_chatbot create-knowledge-base --output-dir=./bioimageio-knowledge-base
```


### Running the BioImage.IO Chatbot in a Docker Container

#### Step 1: Build the Docker Image

To run the BioImage.IO Chatbot using a Docker container, follow these steps. First, build the Docker image by running the following command in your terminal:

```bash
docker build -t bioimageio-chatbot:latest .
```

If you prefer to use a pre-built Docker image from Docker Hub, you can pull the image using the following command:

```bash
docker pull alalulu/bioimageio-chatbot:latest
```


#### Step 2: Start the Chatbot Server

After building the Docker image, you can start the chatbot server with the following command:

```bash
docker run -e OPENAI_API_KEY=sk-xxxxxxxxxxxxx -e BIOIMAGEIO_KNOWLEDGE_BASE_PATH=/knowledge-base -p 3000:9000 -v /path/to/local/knowledge-base:/knowledge-base bioimageio-chatbot:latest python -m bioimageio_chatbot start-server --host=0.0.0.0 --port=9000 --public-base-url=http://localhost:3000
```

Replace the placeholders in the command with the following values:

- `sk-xxxxxxxxxxxxx`: Your OpenAI API key.
- `/path/to/local/knowledge-base`: The local path to your knowledge base folder.

Optionally, for improved reproducibility, you can change `latest` to a version tag such as `v0.1.18`.

#### Step 3: Access the Chatbot

The BioImage.IO Chatbot is now running in the Docker container. You can access it locally in your web browser by visiting:

```
https://bioimage.io/chat?server=http://localhost:3000
```

Make sure to replace `3000` with the host port you specified in the `docker run` command.


Enjoy using the BioImage.IO Chatbot!

## Contribution Process

You can contribute to the chatbot's knowledge base by adding new databases or information.

We use the [`knowledge-base-manifest.yaml`](../knowledge-base-manifest.yaml) file to keep track of the databases and their details.

Follow these steps to contribute to the BioImage.IO Chatbot:

1. Take a look at the [`knowledge-base-manifest.yaml`](../knowledge-base-manifest.yaml) file to see the databases that are currently integrated with the chatbot. The existing data sources are markdown files hosted on github, json files etc.
2. Prepare your database by organising your information to ensure it is accurate, relevant, and structured in a manner that can be easily retrived. You can find some URLs for the existing data sources, please use those as examples.
3. Fork this repository and edit the manifest to include the details of your database, including the name, URL and description.
4. You can submit your contribution with a Pull Request (PR) with the updated manifest. Our team will review and integrate the changes.
5. Once your contribution is accepted and the chatbot's knowledge is updated, test that the chatbot is accurate on its responses when retrieving information from your database.

Remember that, in any step of the process you can contact us to look for feedback or assistance. We deeply appreciate your contribution!

## Developing Chatbot Extensions

### Introduction
The BioImage.IO Chatbot offers a framework designed for easy extensibility, allowing developers to enrich its capabilities with custom extensions. This guide walks you through the process of developing and integrating new extensions into the chatbot, emphasizing the minimal requirements and the steps involved in using ImJoy to interact with the chatbot.

Extensions must expose a callable function that adheres to a specific interface: it should accept a dictionary of parameters as its single argument and return a dictionary containing the results of its operations. This design facilitates seamless integration and communication between the chatbot and its extensions.

A chatbot extension object is a dictionary with the following keys:
 - `id`: a unique identifier for the extension;
 - `name`: the name of the extension;
 - `description`: a short description of the extension;
 - `type`: it must be `bioimageio-chatbot-extension`;
 - `tools`: a dictionary with functions of tools, it represents the set of functions your extension offers, each accepting configuration parameters as input. These functions should carry out specific tasks and return their results in a dictionary;
 - `get_schema`: a function returns the schema for the tools, it returns a JSON schema for each tool function, specifying the structure and types of the expected parameters. This schema is crucial for instructing the chatbot to generate the correct input paramters and validate the inputs and ensuring they adhere to the expected format. Importantly, the chatbot uses the title and description for each field to understand what expected for the tool will generating a function call to run the tool (also see the detailed instructions below).


The following is a chatbot extension object defined in Python:
```python


def get_schema():
    return {
        "my_tool": {
            "type": "object",
            "title": "my_tool",
            "description": "my tool",
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

chatbot_extension = {
    "id": "my-extension",
    "type": "bioimageio-chatbot-extension",
    "name": "My Extension",
    "description": "This is my extension",
    "get_schema": get_schema,
    "tools": {
        "my_tool": my_tool
    }
}
```

Or in JavaScript:
```javascript

const chatbotExtension = {
    id: "my-extension",
    type: "bioimageio-chatbot-extension",
    name: "My Extension",
    description: "This is my extension",
    get_schema() {
        return {
            my_tool: {
                type: "object",
                title: "my_tool",
                description: "my tool",
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
}

```

After creating the extension object, there are two ways to serve the extensions, one is to use the [ImJoy](https://imjoy.io) plugin framework for running extensions in the browser, the other way is to use [Hypha](https://ha.amun.ai) framework to serve the extensions remotely, either in another browser tab or in a native Python process running on your local machine or a remote server.

### Option 1: Register Extension with ImJoy

Below are examples demonstrating how to register an extension with the chatbot using both JavaScript and Python in ImJoy:

You can try them here: https://imjoy.io/lite?plugin=https://if.imjoy.io

#### Register Chatbot Extension with ImJoy in JavaScript

```javascript
const chatbot = await api.createWindow({
    src: "https://bioimage.io/chat",
    name:"BioImage.IO Chatbot",
});
chatbotExtension._rintf = true; // make the chatbot extension as an interface
chatbot.registerExtension(chatbotExtension);
```

#### Register Chatbot Extension with ImJoy in Python

```python
from imjoy_rpc import api

chatbot = await api.createWindow(
    src="https://bioimage.io/chat",
    name="BioImage.IO Chatbot",
)
chatbotExtension._rintf = True # make the chatbot extension as an interface
await chatbot.registerExtension(chatbot_extension)
```

###Option 2: Serve Extension with Hypha

With Hypha, you can serve your extension remotely, enabling seamless integration with the chatbot.

Below are examples demonstrating how to serve an extension with Hypha using both JavaScript and Python:

### Serve Chatbot Extension with Hypha in JavaScript

```javascript
const token = await login({server_url: "https://chat.bioimage.io"})
const server = await connectToServer({server_url: "https://chat.bioimage.io", token});
const svc = await server.registerService(chatbotExtension);
console.log(`Extension service registered with id: ${svc.id}, you can visit the service at: https://bioimage.io/chat?server=${server_url}&extension=${svc.id}`);
```

**IMPORTANT: The above hypha service can only be accessed by the same user who registered the service, below you will find a section about making it public**

#### Serve Chatbot Extension with Hypha in Python

```python
from imjoy_rpc.hypha import connect_to_server, login

server_url = "https://chat.bioimage.io"
token = await login({"server_url": server_url})
server = await connect_to_server({"server_url": server_url, "token": token})
svc = await server.register_service(chatbot_extension)
print(f"Extension service registered with id: {svc.id}, you can visit the service at: https://bioimage.io/chat?server={server_url}&extension={svc.id}")
```

After registering the extension with Hypha, you can access the chatbot with the extension by visiting the following URL: `https://bioimage.io/chat?server=https://chat.bioimage.io&extension=<extension_id>`, where `<extension_id>` is the ID of the registered extension service.

**IMPORTANT: The above hypha service can only be accessed by the same user who registered the service, below you will find a section about making it public**

#### Making Chatbot Extension Public

To make it public, you need to set the visibility of the chatbot extension service to `public`.

See the example below:

```python
from imjoy_rpc.hypha import connect_to_server, login

server_url = "https://chat.bioimage.io"
token = await login({"server_url": server_url})
server = await connect_to_server({"server_url": server_url, "token": token})
# Below, we set the visibility to public
chatbot_extension['config'] = {"visibility": "public"}
svc = await server.register_service(chatbot_extension)
print(f"Extension service registered with id: {svc.id}, you can visit the service at: https://bioimage.io/chat?server={server_url}&extension={svc.id}")
```

You can also implement authorization logic in the tool function, see [hypha service authorization](https://ha.amun.ai/#/?id=service-authorization).

### Tutorial

For an in-depth understanding, refer to [our detailed tutorial](./docs/bioimage-chatbot-extension-tutorial.ipynb), accessible directly through the ImJoy Jupyter Notebook in your browser without installation. [Click here to launch the notebook](https://imjoy-notebook.netlify.app/lab/index.html?load=https://raw.githubusercontent.com/bioimage-io/bioimageio-chatbot/main/docs/bioimage-chatbot-extension-tutorial.ipynb&open=1).

### Extension Development Details

#### `tools` and `get_schema`
When developing extensions, it's essential to define the `tools` and `get_schema` functionalities carefully:
- **`tools`**: Represents the set of functions your extension offers, each accepting configuration parameters as input. These functions should carry out specific tasks and return their results in a dictionary.
- **`get_schema`**: Returns a JSON schema for each tool function, specifying the structure and types of the expected parameters. This schema is crucial for instructing the chatbot to generate the correct input paramters and validate the inputs and ensuring they adhere to the expected format. Importantly, the chatbot uses the title and description for each field to understand what expected for the tool will generating a function call to run the tool (also see the detailed instructions below).

#### Notes on Function Input/Output
The input and output of tool functions are restricted to primitive types (e.g., numbers, strings) that can be encoded in JSON. This limitation ensures compatibility and facilitates the exchange of data between the chatbot and extensions.

#### Importance of Detailed Descriptions
Providing a detailed description for your extension and its arguments is vital. These descriptions assist the chatbot in correctly invoking the tools and help the chatbot understand the functionality and purpose of your extension. Ensure that each argument is accompanied by a clear title and a comprehensive description to improve usability and interaction quality of the chatbot.

By adhering to these guidelines, you will enhance the clarity, utility, and ease of integration of your chatbot extensions, contributing to a richer ecosystem of tools within the BioImage.IO community.

