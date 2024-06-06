# Installation Guide

## Setup the Chatbot locally

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

