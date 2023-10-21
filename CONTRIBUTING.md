# BioImage.IO Chatbot Contribution Guidelines

Thank you for your interest in contributing to the BioImage.IO Chatbot. Your contributions help us enhance the chatbot's knowledge base and provide more accurate and detailed responses. This document outlines how you can contribute new databases or information for retrieval by the chatbot.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Database Requirements](#database-requirements)
- [Contribution Process](#contribution-process)
- [Contact Us](#contact-us)

## Introduction
The BioImage.IO Chatbot relies on a comprehensive knowledge base to provide accurate responses to user queries. We encourage contributions to expand this knowledge base by adding new databases, information, or resources. Whether you're a researcher, developer, or domain expert, your contributions can help improve the chatbot's functionality.

## Installation
Before you begin the contribution process, you need to set up the environment and run the necessary scripts. Follow these steps to get started:

1. Install Dependencies: Install the required Python dependencies by running the following commands:
```pip install -r requirements.txt```

2. Create Knowledge Base: Execute the knowledge base creation script:
```python create-knowledge-base.py```

3. Start the Chatbot: Launch the BioImage.IO Chatbot using the following command:
```python start-bioimageio-chatbot.py```

*Note: You may need to set the OPENAI_API_KEY environment variable to run the chatbot.*

## Database Requirements
Before contributing, make sure your database meets the following requirements:

* Relevance: The information should be related to bioimage analysis, software tools, models, or datasets.
* Accuracy: Information should be accurate and up-to-date.
* Structured Data: Data should be organized in a clear and structured manner, making it easy to retrieve. Remember to always have clear definitions and to give context about what are you doing so the information is easier to retrieve.

## Contribution Process
Follow these steps to contribute to the BioImage.IO Chatbot:
1. Download the chatbot's [manifest file](manifest.yaml), which lists existing databases and their details.
2. Prepare your database by organising your information to ensure it is accurate, relevant, and structured in a manner that can be easily retrived. See the database requirements above. 
3. Edit the manifest to include the details of your database, including the name, URL and description.
4. You can submit your contribution with a Pull Request (PR) with the updated manifest. Our team will review and integrate the changes.
5. Once your contribution is accepted and the chatbot's knowledge is updated, test that the chatbot is accurate on its responses when retrieving information from your database.

Remember that, in any step of the process you can contact us to look for feedback or assistance. We deeply appreciate your contribution!

## Contact Us
If you have any questions, need assistance, or want to contribute to the chatbot's knowledge base, please don't hesitate to contact us. Our team is here to help you get started and make valuable contributions.

Thank you for your support and for helping make the BioImage.IO Chatbot more informative and useful to the community.
> missing email or similar