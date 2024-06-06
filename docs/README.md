# 🦒 BioImage.IO Chatbot 🤖

**📣New Preprint: [![arXiv](https://img.shields.io/badge/arXiv-2310.18351-red.svg)](https://arxiv.org/abs/2310.18351) <a href="https://zenodo.org/records/10032227" target="_blank"><img id="record-doi-badge" data-target="[data-modal='10.5281/zenodo.10032228']" src="https://zenodo.org/badge/DOI/10.5281/zenodo.10032228.svg" alt="10.5281/zenodo.10032228"></a>**

**📚 Documentation: https://bioimage-io.github.io/bioimageio-chatbot/**

**👇 Want to Try the Chatbot? [Visit here!](https://bioimage.io/chat)**

## Your Personal Assistant in Computational Bioimaging

Welcome to the BioImage.IO Chatbot user guide. This guide will help you get the most out of the chatbot, providing detailed information on how to interact with it and retrieve valuable insights related to computational bioimaging.

## Introduction

The BioImage.IO Chatbot is a versatile conversational agent designed to assist users in accessing information related to computational bioimaging. It leverages the power of Large Language Models (LLMs) and integrates user-specific data to provide contextually accurate and personalized responses. Whether you're a researcher, developer, or scientist, the chatbot is here to make your bioimaging journey smoother and more informative.


![screenshot for the chatbot](./docs/screenshots/chatbot-animation.gif)

The following diagram shows how the chatbot works:

<img src="https://docs.google.com/drawings/d/e/2PACX-1vROHmf1aZPMLOMvwjot1laB9wvRsaDkjkYbGNNveqN-Pm_9xWlD48krQMobWT1WrrOrZnwH9gPLsDRw/pub?w=1392&amp;h=1112">

## Chatbot Features

The BioImage.IO Chatbot is equipped with an array of capabilities designed to enhance the bioimaging experience:

* **Contextual and Personalized Response**: Interprets the context of inquiries to deliver relevant and accurate responses. Adapts interactions based on user-specific background information to provide customized advice.

* **Comprehensive Data Source Integration**: Accesses a broad range of databases and documentation for bioimaging, including [bio.tools](https://bio.tools), [ImageJ.net](https://imagej.net/), [deepImageJ](https://deepimagej.github.io/deepimagej/), [ImJoy](https://imjoy.io/#/), and [bioimage.io](https://bioimage.io). Details on the supported sources are maintained in the [`knowledge-base-manifest.yaml`](knowledge-base-manifest.yaml) file.

* **Advanced Query Capabilities**: Generates and executes Python scripts for detailed queries within structured databases such as CSV, JSON files, or SQL databases, facilitating complex data retrievals.

* **AI-Powered Analysis and Code Interpretation**: Directly runs complex image analysis tasks using advanced AI models like Cellpose, via an embedded code interpreter.

* **Performance Enhancements with ReAct and RAG**: Utilizes a Retrieval Augmented Generation system with a ReAct loop for dynamic, iterative reasoning and tool engagement, improving response quality.

* **Extension Mechanism for Developers**: Allows for the development of custom extensions using ImJoy plugins or hypha services within Jupyter notebooks, enhancing flexibility and integration possibilities.

* **Vision Inspection and Hardware Control**: Features a Vision Inspector extension powered by GPT-4 for visual feedback on image content and analysis outcomes, and demonstrates potential for controlling microscopy hardware in smart microscopy setups.

* **Interactive User Interface and Documentation**: Offers a user-friendly interface with comprehensive support documents, ensuring easy access to its features and maximizing user engagement.

## Using the Chatbot

We are providing a public chatbot service for you to try out. You can access the chatbot [here](https://chat.bioimage.io/chat).

Please note that the chatbot is still in beta and is being actively developed, we will log the message you input into the chatbot for further investigation of issues and support our development. See the [Disclaimer for BioImage.IO Chatbot](./docs/DISCLAIMER.md). If you want to to remove your chat logs, please contact us via [this form](https://oeway.typeform.com/to/K3j2tJt7).

Here you can find usage guide and more examples: [Usage guide and example screenshots](docs/usage-example.md).

If you encounter any issues, please report them via [Github](https://github.com/bioimage-io/bioimageio-chatbot/issues).


### Asking Questions

To ask the chatbot a question, type your query and send it. The chatbot will analyze your question and provide a relevant response. You can ask questions related to bioimaging, software tools, models, and more.

### Personalized Responses

The chatbot uses your user profile information, such as your name, occupation, and background, to personalize its responses. This ensures that the information you receive is tailored to your specific needs.


## Setup Your Own Chatbot

You can also set up your own chatbot server. Please refer to the [installation guide](./docs/installation.md) for detailed instructions on how to set up the chatbot server on your local machine or server.

## Technical Overview

Please read the [technical overview](./docs/technical-overview.md) for more details about the chatbot's design and implementation.

## Develop Chatbot Extensions

The BioImage.IO Chatbot is designed to be extensible, allowing developers to create custom extensions to add new functionalities to the chatbot. You can create extensions to integrate new tools, databases, and services into the chatbot, making it more powerful and versatile. See the [development guide](./docs/development.md) for more details.

## Join Us as a Community Partner

The BioImage.IO Chatbot is a community-driven project. We welcome contributions from the community to help improve the chatbot's knowledge base and make it more informative and useful to the community.

For more information, please visit the [contribution guidelines](docs/CONTRIBUTING.md).

If you are a tool developer or a database maintainer related to bioimaging, you can join us as a community partner. Please get in touch with us via [Github issues](https://github.com/bioimage-io/bioimageio-chatbot/issues).

## Contact Us

If you have any questions, need assistance, or want to contribute to the chatbot's knowledge base, please do not hesitate to contact us via [Github issues](https://github.com/bioimage-io/bioimageio-chatbot/issues). Our team is here to help you get started and make valuable contributions.

Thanks for your support and helping make the BioImage.IO Chatbot more informative and useful to the community.

## Publication

For detailed description of our work, please read our preprint: **[![arXiv](https://img.shields.io/badge/arXiv-2310.18351-red.svg)](https://arxiv.org/abs/2310.18351) <a href="https://zenodo.org/records/10032227" target="_blank"><img id="record-doi-badge" data-target="[data-modal='10.5281/zenodo.10032227']" src="https://zenodo.org/badge/DOI/10.5281/zenodo.10032227.svg" alt="10.5281/zenodo.10032227"></a>**


To reproduce the use cases described in [Figure 2](https://docs.google.com/drawings/d/e/2PACX-1vTIRwRldQBnTFqz0hvS01znGOEdoeDMJmZC-PlBM-O59u_xo7DfJlUEE9SlRsy6xO1hT2HuSOBrLmUz/pub?w=1324&amp;h=1063) in the manuscript, please refer to the [reproducing example usage scenarios](./docs/figure-2-use-cases.md).

<img style="width:300px;" src="https://docs.google.com/drawings/d/e/2PACX-1vTIRwRldQBnTFqz0hvS01znGOEdoeDMJmZC-PlBM-O59u_xo7DfJlUEE9SlRsy6xO1hT2HuSOBrLmUz/pub?w=1324&amp;h=1063">


## Cite Us

If you use the BioImage.IO Chatbot in your research, please cite us:

```
Lei, W., Fuster-Barceló, C., Muñoz-Barrutia, A., & Ouyang, W. (2023). 🦒BioImage.IO Chatbot: A Personalized Assistant for BioImage Analysis Augmented by Community Knowledge Base (0.1.0). Zenodo. https://doi.org/10.5281/zenodo.10032228
```

## Acknowledgements

We thank [AI4Life consortium](https://ai4life.eurobioimaging.eu/) for its crucial support in the development of the BioImage.IO Chatbot.

![AI4Life](https://ai4life.eurobioimaging.eu/wp-content/uploads/2022/09/AI4Life-logo_giraffe-nodes-2048x946.png)

AI4Life has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement number 101057970. Views and opinions expressed are, however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.
