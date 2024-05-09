# 🦒 BioImage.IO Chatbot 🤖

**📣New Preprint: [![arXiv](https://img.shields.io/badge/arXiv-2310.18351-red.svg)](https://arxiv.org/abs/2310.18351) <a href="https://zenodo.org/records/10032227" target="_blank"><img id="record-doi-badge" data-target="[data-modal='10.5281/zenodo.10032228']" src="https://zenodo.org/badge/DOI/10.5281/zenodo.10032228.svg" alt="10.5281/zenodo.10032228"></a>**

**👇 Want to Try the Chatbot? [Visit here!](https://bioimage.io/chat)**

## Your Personal Assistant in Computational Bioimaging

Welcome to the BioImage.IO Chatbot user guide. This guide will help you get the most out of the chatbot, providing detailed information on how to interact with it and retrieve valuable insights related to computational bioimaging.

## Contents
- [Introduction](#introduction)
- [Chatbot Features](#chatbot-features)
- [Using the Chatbot](#using-the-chatbot)
- [Assistants](#assistants)
- [Example Use cases](#example-use-cases)
- [Accessing the Chatbot (GPTs)](#accessing-the-chatbot-gpts)
- [Contact Us](#contact-us)
- [Cite Us](#cite-us)
- [Disclaimer](#disclaimer)

## Introduction

The BioImage.IO Chatbot is a versatile conversational agent designed to assist users in accessing information related to computational bioimaging. It leverages the power of Large Language Models (LLMs) and integrates user-specific data to provide contextually accurate and personalized responses. Whether you're a researcher, developer, or scientist, the chatbot is here to make your bioimaging journey smoother and more informative.

![screenshot for the chatbot](./docs/screenshots/chatbot-animation.gif)

The following diagram shows how the chatbot works:

<img src="https://docs.google.com/drawings/d/e/2PACX-1vROHmf1aZPMLOMvwjot1laB9wvRsaDkjkYbGNNveqN-Pm_9xWlD48krQMobWT1WrrOrZnwH9gPLsDRw/pub?w=1392&amp;h=1112">

## Chatbot Features

The BioImage.IO Chatbot is equipped with an array of capabilities designed to enhance the bioimaging experience:

- **Contextual and Personalized Response**: Interprets the context of inquiries to deliver relevant and accurate responses. Adapts interactions based on user-specific background information to provide customized advice.

- **Comprehensive Data Source Integration**: Accesses a broad range of databases and documentation for bioimaging, including [bio.tools](https://bio.tools), [ImageJ.net](https://imagej.net/), [deepImageJ](https://deepimagej.github.io/deepimagej/), [ImJoy](https://imjoy.io/#/), and [bioimage.io](https://bioimage.io). Details on the supported sources are maintained in the [`knowledge-base-manifest.yaml`](knowledge-base-manifest.yaml) file.

- **Advanced Query Capabilities**: Generates and executes Python scripts for detailed queries within structured databases such as CSV, JSON files, or SQL databases, facilitating complex data retrievals.

- **AI-Powered Analysis and Code Interpretation**: Directly runs complex image analysis tasks using advanced AI models like Cellpose, via an embedded code interpreter.

- **Performance Enhancements with ReAct and RAG**: Utilizes a Retrieval Augmented Generation system with a ReAct loop for dynamic, iterative reasoning and tool engagement, improving response quality.

- **Extension Mechanism for Developers**: Allows for the development of custom extensions using ImJoy plugins or hypha services within Jupyter notebooks, enhancing flexibility and integration possibilities.

- **Vision Inspection and Hardware Control**: Features a Vision Inspector extension powered by GPT-4 for visual feedback on image content and analysis outcomes, and demonstrates potential for controlling microscopy hardware in smart microscopy setups.

- **Interactive User Interface and Documentation**: Offers a user-friendly interface with comprehensive support documents, ensuring easy access to its features and maximizing user engagement.

## Using the Chatbot

We are providing a public chatbot service for you to try out. You can access the chatbot [here](https://chat.bioimage.io/chat).

Please note that the chatbot is still in beta and is being actively developed, we will log the message you input into the chatbot for further investigation of issues and support our development. See the [Disclaimer for BioImage.IO Chatbot](./docs/DISCLAIMER.md). If you want to to remove your chat logs, please contact us via [this form](https://oeway.typeform.com/to/K3j2tJt7).

### Assistants
- **Melman**: An assistant specializing in searching through bioimage analysis tools, including the BioImage Model Zoo, the BioImage Archive, biii.eu, and others, to provide comprehensive support in bioimaging.
- **Nina**: Your tutor for learning about bioimage analysis and AI. Nina is the perfect guide, offering detailed explanations and support throughout your learning journey.
- **Bridget**: Offers tools to analyze your images directly or helps you construct a Python pipeline for advanced image processing tasks.
- **Skyler**: Facilitates the integration of bioimaging tools and workflows and extensions, making it easier to apply advanced analysis techniques in your research.

### Example Use cases

#### Melman: querying bioimage analysis documentation
Querying the BioImage Model Zoo:
- **Initiate a Query**: Type a question related to bioimage analysis, e.g., "What are the best practices for optimizing model performance on bioimage.io?"
- **Review the Chatbot's Response**: The chatbot will provide an answer that includes information extracted from the BioImage Model Zoo documentation.

Exploring the Human Protein Atlas:
- **Initiate a Query**: Ask the chatbot to find protein information in the Human Protein Atlas by typing "Tell me about PML protein and show me the cell images"
- **Interpret the Results**: The chatbot will respond by constructing an API call to the Protein Atlas database and displaying the relevant information about the PML protein, including cell images.

Querying the BioImage Archive
- **Initiate a Query**: Ask the chatbot to find cell images at the G1 phase by typing "Please, find datasets of cell images at G1 phase."
- **Interpret the Results**: The chatbot will initiate an API call to the BioImage Archive server, and return results such as a study titled "DeepCycle: Deep learning reconstruction of the closed cell cycle trajectory from single-cell unsegmented microscopy images."

#### Nina: Learning about Bioimage Analysis and AI
- **Initiate a Query**: Ask the chatbot to explain the concept of "image segmentation" or "deep learning in bioimage analysis".
- **Interpret the Results**: The chatbot will provide detailed explanations and examples to help you understand the concepts better.

#### Bridget: Running AI Models for Image Analysis

- **Prereqsitues**: Ensure you have Chrome or a Chromium-based browser installed on your computer.
- **Download Image Data**: Begin by creating a new folder on your computer named `test-images`. Download the image data file from [this link](https://raw.githubusercontent.com/bioimage-io/bioimageio-chatbot/main/docs/example-data/nuclei.tif) and save it into the `test-images` folder.
- **Initiate Bioimage Analyst**: Navigate to the BioImage.IO chatbot interface at https://bioimage.io/chat/. Note that only Chrome or Chromium-based browser is supported at the moment. Select "Bioimage Analyst(Bridget)" located in the upper right corner of the chatbot interface.
- **Mount your Data Folder**: Within the chat interface, click on the "Mount Files" button located below the dialog window. This action will allow you to mount the test-image folder that contains your downloaded image data. The chatbot will confirm the successful mounting of the folder, you can now ask it to list the files contained within, and ensuring that your data is ready for analysis.
- **Perform segmentation using Cellpose model**: Type "Segment the image `/mnt/nuclei.tif` using Cellpose" to run the Cellpose model on the image data. Upon successful execution of the model, the chatbot will notify you that the segmentation process is complete and will display the analyzed results. Optionally, you can ask it to "count the number of nuclei in the image" if successfully segmented, "plot the size distribution of nuclei", or you can tell it to "use the visual inspection tool to analyze the figure and create a report about the size distribution".

#### Skyler: Developing New Extensions

One extension example is to develop a new extension for controlling a Microscope Stage and Capturing Images.

- **Pre-requisites**: You will need a microscope and the squid control software
- **Create microscope extension**: Following the example in the above [chatbot extension example notebook](https://imjoy-notebook.netlify.app/lab/index.html?load=https://raw.githubusercontent.com/bioimage-io/bioimageio-chatbot/main/docs/bioimage-chatbot-extension-tutorial.ipynb&open=1), create a hypha service extension for controlling the microscope:
    1. **Setup the Developer Environment**: Open a Jupyter Notebook. Install and import the `imjoy_rpc` and `pydantic` packages.
    2. **Define Input Schemas**: Create classes for `MoveStageInput` and `SnapImageInput` to structure the user input. (Note: To help the chatbot understand the "center", you will need to tell the chatbot about the boundaries of the stage via the docstring of the `MoveStageInput` class)
    3. **Implement Control Functions**: Write asynchronous functions `move_stage` and `snap_image`.
    4. **Setup Extension Interface**: Develop the extension interface and define a schema getter function.
    5. **Register the Extension**: Register the extension as hypha server and connect to the the chatbot.
- **Initiate a Query**: Ask the chatbot to "Please move to the center and snap an image".
- **Interpret the Results**: The chatbot will execute the `move_stage` function to move the microscope stage to the center and then capture an image using the `snap_image` function. The chatbot will confirm the successful completion of the tasks.

More details on how to develop new extensions can be found in the developers documentation. 

## Accessing the Chatbot (GPTs)
In addition to standalone usage, the BioImage.IO Chatbot supports porting extensions to OpenAI custom [GPTs](https://chat.openai.com/gpts) for users with OpenAI accounts. Chatbot extensions following the development model specified in the [development guidelines](./development.md) and [notebook tutorial](./bioimage-chatbot-extension-tutorial.ipynb) are automatically converted to `openapi` schema which can be used to create OpenAI GPTs using the online GPT creator. 

`openapi` schemas for extensions are generated on Chatbot server startup via the `register_service` function in  [gpts_action.py](../bioimageio_chatbot/gpts_action.py). These schemas are then made available for OpenAI GPT creator import directly via url. This process for creating a custom GPT from the public BioImage.IO Chatbot instance extensions is shown below. Users are encouraged to submit their extensions to the BioImage.IO team for incorporation into the public Chatbot instance. 

Note that GPT actions are run through the hosted server instance (chat.bioimage.io in the case of the public Chatbot instance). Also note that the creation of custom OpenAI GPTs requires a paid OpenAI account. 

### Creating a Custom GPT from the public Chatbot Instance
The public Chatbot instance's `openapi` extension schema are available at the following link: `https://chat.bioimage.io/public/services/bioimageio-chatbot-extensions-api/get_openapi_schema`

After logging in to their OpenAI accounts, users can navigate to the GPTs [page](https://chat.openai.com/gpts) and click `Create` as shown below:

![gpt_landing_page](./docs/screenshots/gpts_landing_page.png)

To add GPT actions from Chatbot extensions, navigate to the `Configure` tab and select `Create new action`:

![configure_gpt](./docs/screenshots/configure_gpt.png)

The Chatbot-generated `openapi` schema can then be imported direct by selecting `Import from URL` and inputting the public Chatbot's extension schema `https://chat.bioimage.io/public/services/bioimageio-chatbot-extensions-api/get_openapi_schema`

Users can edit the JSON content to select individual actions from the Chatbot extensions if desired:

![gpts_json](./docs/screenshots/gpts_json.png)

## Contact Us
If you have any questions, need assistance, or want to contribute to the chatbot's knowledge base, please don't hesitate to contact us via [Github issues](https://github.com/bioimage-io/bioimageio-chatbot/issues). Our team is here to help you get started and make valuable contributions.

## Cite Us
If you find the BioImage.IO Chatbot useful for your research or work, please consider citing our work:

```bibtex
@article{lei2023bioimage,
  title={BioImage. IO Chatbot: A Personalized Assistant for BioImage Analysis Augmented by Community Knowledge Base},
  author={Lei, Wanlu and Fuster-Barcel{\'o}, Caterina and Mu{\~n}oz-Barrutia, Arrate and Ouyang, Wei},
  journal={arXiv preprint arXiv:2310.18351},
  year={2023}
}
```

## Disclaimer
### General Usage
The BioImage.IO Chatbot ("Chatbot") is intended for informational purposes only and aims to assist users in navigating the resources, tools, and workflows related to bioimage analysis. While we strive for accuracy, the Chatbot is not a substitute for professional advice, consultation, diagnosis, or any kind of formal scientific interpretation.

### No Warranties
The Chatbot service is provided "as is" and "as available" without any warranties of any kind, either express or implied, including but not limited to the implied warranties of merchantability, fitness for a particular purpose, or non-infringement. We make no warranty that the service will meet your requirements or be available on an uninterrupted, secure, or error-free basis.

### Liability
Under no circumstances will we be liable for any loss or damage incurred as a result of the use of this Chatbot, including but not limited to any errors or omissions in the content, any unauthorized access to or use of our servers, or any loss of data or profits.

### User Responsibility
The user assumes all responsibility and risk for the use of this Chatbot. It is the user's responsibility to evaluate the accuracy, completeness, or usefulness of any information, opinion, or content available through the Chatbot service.

### Third-Party Links
The Chatbot may provide links to external websites or resources for your convenience. We have no control over these sites and resources, and we are not responsible for their availability, reliability, or the content provided.

### Data Privacy
User interactions with the Chatbot may be stored for analysis and improvement of the service. All data will be handled in accordance with our Privacy Policy.

### Privacy Policy
The personal data you may provide will be used to disseminate information pertaining to the execution of the Horizon Europe Funded AI4Life project (Grant number: 101057970). In accordance with the Grant Agreement, your data will be retained during the project and deleted when it has ended as soon as the retention period established by the EC is over. If you would like to update or delete your data during the course of the project, please contact us using [this form](https://oeway.typeform.com/to/K3j2tJt7?typeform-source=bioimage.io).

### Modifications
We reserve the right to modify this disclaimer at any time, effective upon posting of an updated version on this website. Continued use of the Chatbot after any such changes shall constitute your consent to such changes.
