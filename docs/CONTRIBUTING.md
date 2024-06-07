# BioImage.IO Chatbot Contribution Guidelines

Thank you for your interest in contributing to the BioImage.IO Chatbot. Your contributions help us enhance the chatbot's knowledge base and provide more accurate and detailed responses. This document outlines how you can contribute new databases or information for retrieval by the chatbot.

## Introduction

The BioImage.IO Chatbot relies on a comprehensive knowledge base to provide accurate responses to user queries. We encourage contributions to expand this knowledge base by adding new databases, information, or resources. Whether you're a researcher, developer, or domain expert, your contributions can help improve the chatbot's functionality.

## Contribution Process
### Knowledge Base

You can contribute to the chatbot's knowledge base by adding new databases or information.

We use the [`knowledge-base-manifest.yaml`](../knowledge-base-manifest.yaml) file to keep track of the databases and their details.

Follow these steps to contribute to the BioImage.IO Chatbot:

1. Take a look at the [`knowledge-base-manifest.yaml`](../knowledge-base-manifest.yaml) file to see the databases that are currently integrated with the chatbot. The existing data sources are markdown files hosted on github, json files etc.
2. Prepare your database by organising your information to ensure it is accurate, relevant, and structured in a manner that can be easily retrived. You can find some URLs for the existing data sources, please use those as examples.
3. Fork this repository and edit the manifest to include the details of your database, including the name, URL and description.
4. You can submit your contribution with a Pull Request (PR) with the updated manifest. Our team will review and integrate the changes.
5. Once your contribution is accepted and the chatbot's knowledge is updated, test that the chatbot is accurate on its responses when retrieving information from your database.

Remember that, in any step of the process you can contact us to look for feedback or assistance. We deeply appreciate your contribution!

### Develop Custom Extenstion

The BioImage.IO Chatbot offers a framework designed for easy extensibility, allowing developers to enrich its capabilities with custom extensions. Please check details on how to contribute to the chatbot by developing custom extension [`Developing Chatbot Extensions`](./development.md).


## Contact Us

If you have any questions, need assistance, or want to contribute to the chatbot's knowledge base, please don't hesitate to contact us via [Github issues](https://github.com/bioimage-io/bioimageio-chatbot/issues). Our team is here to help you get started and make valuable contributions.
