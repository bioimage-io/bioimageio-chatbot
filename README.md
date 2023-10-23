# 🤖 BioImage.IO ChatBot

## Your Personal Assistant in BioImage Analysis

Welcome to the BioImage.IO Chatbot user guide. This guide will help you get the most out of the chatbot, providing detailed information on how to interact with it and retrieve valuable insights related to bioimage analysis.

## Introduction

The BioImage.IO Chatbot is a versatile conversational agent designed to assist users in accessing information related to bioimage analysis. It leverages the power of Large Language Models (LLMs) and integrates user-specific data to provide contextually accurate and personalized responses. Whether you're a researcher, developer, or scientist, the chatbot is here to make your bioimage analysis journey smoother and more informative.

## Chatbot Features

The BioImage.IO Chatbot offers the following features:

* **Contextual Understanding**: The chatbot can understand the context of your questions, ensuring responses are relevant and informative.

* **Personalization**: By incorporating your background information, the chatbot tailors responses to meet your specific requirements.

* **Document Retrieval**: It can search through extensive documentation to provide detailed information on models, applications, datasets, and more. For example, the ChatBot is able to retrieve information from the [bio.tools](https://bio.tools), [ImageJ.net] (https://imagej.net/), [deepImageJ](https://deepimagej.github.io), [ImJoy](https://imjoy.io/#/) and [bioimage.io](https://bioimage.io). The full list of supported databases can be found in the [`knowledge-base-manifest.yaml`](knowledge-base-manifest.yaml) file.

* **Query Structured Database by Script Execution**: The chatbot can generate Python scripts for complex queries in structured databases (e.g., csv, json file, SQL databases), helping you perform advanced tasks such as specific questions about the available models at [bioimage.io](https://bioimage.io).

## Using the Chatbot

You can visit the BioImage.IO Chatbot at [https://chat.bioimage.io](https://chat.bioimage.io)[TBD: Available Soon]. Please note that the chatbot is still in beta and is being actively developed. See the [Disclaimer for BioImage.IO Chatbot](./docs/DISCLAIMER.md).

If you encounter any issues, please report them via [Github](https://github.com/bioimage-io/bioimageio-chatbot/issues).


### Asking Questions

To ask the chatbot a question, type your query and send it. The chatbot will analyze your question and provide a relevant response. You can ask questions related to bioimage analysis, software tools, models, and more.

### Personalized Responses

The chatbot uses your user profile information, such as your name, occupation, and background, to personalize its responses. This ensures that the information you receive is tailored to your specific needs.


## Setup the Chatbot locally

If you want to run the chatbot server locally, you need to have an OpenAI API key. You can get one by signing up at [OpenAI](https://beta.openai.com/). Once you have your API key, you can install the chatbot package via pip and set the environment variables:

```bash
pip install bioimageio-chatbot
```

```bash
export OPENAI_API_KEY=sk-xxxxxxxx # Required
export BIOIMAGEIO_KNOWLEDGE_BASE_PATH=/path/to/bioimageio-knowledge-base  # Optional, default to ./bioimageio-knowledge-base 
```

### Command-line Interface

BioImage.IO Chatbot comes with a command-line interface to facilitate server management, connection to external servers, and knowledge base creation.

You can access the command-line interface by running `python -m bioimageio_chatbot` or the `bioimageio-chatbot` command.

Below are the available commands and options:

### Initialize Knowledge Base

To initialize the knowledge base, use the `init` command:

```bash
python -m bioimageio_chatbot init
```

This will load the knowledge base from the location specified by the `BIOIMAGEIO_KNOWLEDGE_BASE_PATH` environment variable, or use the default path `./bioimageio-knowledge-base`. If the knowledge base is not found, it will be downloaded from the predefined URL (by default, it uses https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimageio-knowledge-base. It can be override with `BIOIMAGEIO_KNOWLEDGE_BASE_URL`).

**Example:**

```bash
export BIOIMAGEIO_KNOWLEDGE_BASE_PATH="./my_knowledge_base"
python -m bioimageio_chatbot init
```

After running the `init` command, it will list the databases that are loaded into the knowledge base.

#### Start Server

To start your own server entirely, use the `start-server` command:

```bash
python -m bioimageio_chatbot start-server [--host HOST] [--port PORT] [--public-base-url PUBLIC_BASE_URL]
```

**Options:**

- `--host`: The host address to run the server on (default: `0.0.0.0`)
- `--port`: The port number to run the server on (default: `9000`)
- `--public-base-url`: The public base URL of the server (default: `http://127.0.0.1:9000`)

**Example:**

```bash
export OPENAI_API_KEY=sk-xxxxxxxx
export BIOIMAGEIO_KNOWLEDGE_BASE_PATH=./bioimageio-knowledge-base
python -m bioimageio_chatbot start-server --host=0.0.0.0 --port=9000
```
This will create a local server, and the BioImage.IO Chatbot is available at: http://127.0.0.1:9000/public/apps/bioimageio-chatbot-client/index

Open the link in a browser and you will see:
![screenshot for the chatbot](./docs/screenshot-hi.png)

Please note that the chatbot server may not be accessible to users outside your local network.

To be able to share your chatbot service over the internet (especially for users outside your local network), you will need to expose your server publicly. Otherwise, please see [Connect to Server](#connect-to-server)

#### Connect to Server

To help you share your chatbot with users external to your local network, you can use our public [BioEngine](https://aicell.io/project/bioengine/) server as a proxy.

To connect to an external BioEngine server, use the `connect-server` command:

```bash
python -m bioimageio_chatbot connect-server [--server-url SERVER_URL]
```

**Options:**

- `--server-url`: The URL of the external BioEngine server to connect to (default: `https://ai.imjoy.io`)

**Example:**

```bash
export OPENAI_API_KEY=sk-xxxxxxxx
export BIOIMAGEIO_KNOWLEDGE_BASE_PATH=./bioimageio-knowledge-base
python -m bioimageio_chatbot connect-server --server-url=https://ai.imjoy.io
```

First, you will be asked to log in with a hypha account. Either your github or google account can be reused. Then, the following message containing a link to the chatbot will be displayed: 'The BioImage.IO Chatbot is available at: https://ai.imjoy.io/github|xxxxxx/apps/bioimageio-chatbot-client/index'

Leave your chatbot running to enable users inside or outside your network to access it from this URL.

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

## Join Us as a Community Partner

The BioImage.IO Chatbot is a community-driven project. We welcome contributions from the community to help improve the chatbot's knowledge base and make it more informative and useful to the community.

For more information, please visit the [contribution guidelines](docs/CONTRIBUTING.md).

If you are a tool developer or a database maintainer related to bioimage analysis, you can join us as a community partner. Please contact us via [Github issues](https://github.com/bioimage-io/bioimageio-chatbot/issues).

## Contact Us

If you have any questions, need assistance, or want to contribute to the chatbot's knowledge base, please do not hesitate to contact us via [Github issues](https://github.com/bioimage-io/bioimageio-chatbot/issues). Our team is here to help you get started and make valuable contributions.

Thank you for your support and for helping make the BioImage.IO Chatbot more informative and useful to the community.

## Acknowledgements

We thank [AI4Life consortium](https://ai4life.eurobioimaging.eu/) for its crucial support in the development of the BioImage.IO Chatbot.

![AI4Life](https://ai4life.eurobioimaging.eu/wp-content/uploads/2022/09/AI4Life-logo_giraffe-nodes-2048x946.png)

AI4Life has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement number 101057970. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.
