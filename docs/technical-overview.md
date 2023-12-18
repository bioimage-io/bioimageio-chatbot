# Design and Functionality of BioImage.IO Chatbot: A User Guide and Technical Overview

## Chatbot Interface

After following the installation guidelines from the [README](/README.md), the chat interface will resemble Figure 1.

![BioImage.IO-Chatbot](./screenshots/chat-interface.png)
*Figure 1. The chat interface of the BioImage.IO Chatbot.*

Users can input their profiles as depicted in Figure 2. 
![user-profile](./screenshots/user-profile.png)
*Figure 2. Users can personalize responses by clicking `Edit Profile` and save their settings for future conversations by clicking `Save`.*

As of today, our chatbot utilizes knowledge bases from the following pivotal communities: bioimage.io [2], Imjoy [3], deepimageJ [4], ImageJ [5], bio.tools [6], and scikit-image [7]. A key feature allows users to specify a preferred channel for information retrieval, as shown in Figure 3. If a channel is designated, the chatbot sources information from that specific community. Otherwise, it uses an intelligent selection process driven by a schema-based agent to choose the most relevant channel based on the user's query.

![channels](./screenshots/channels.png)
*Figure 3. Users can personalize the conversation by selecting a specific channel from the ‘Knowledge Base Channel’.*

### Building the Knowledge Base

The knowledge base is efficiently and collaboratively constructed by downloading documentation from given URLs. These can be repositories, PDFs, or other forms of documentation. We use a regular expression splitter to segment the documentation into manageable chunks for efficient and accurate retrieval. These chunks are then embedded and stored as vectors in a FAISS [1]-based vector database.

## Schema-Based Agent Design

The chatbot's ability to understand and respond to user queries is substantially improved by employing a schema-based agent design. Unlike traditional context-based models, our approach utilizes predefined schemas to guide the conversation and information retrieval process. 

The schema-based agent operates on the function-call LLM [8], and uses input and output schemas to generate text output. Within this implementation, we construct a customer service chatbot by defining a role class, as shown in Figure 4.

![role_create](./screenshots/role_create.png)
*Figure 4. Creation of a chatbot role class named ‘CustomerServiceRole’ by defining fields of the role class.*

## Response Modes
The BioImage.IO Chatbot employs diverse methods to generate responses, currently encompassing five distinct response modes. The response mode is chosen by the schema-based agent based on the user's query and the selected channel.

### Direct Response
This mode delivers concise, direct answers for straightforward queries, utilizing the chatbot's internal knowledge base without external data retrieval. 

    ![direct-response](./screenshots/direct-response.png)
    *Figure 6. Direct response example.*

### Document Retrieval Response
Complex or specific queries trigger this mode, where the chatbot extracts essential elements from the user's question to fetch information from the relevant documentation. 

    ![retrieval-text](./screenshots/retrieval-text.png)
    *Figure 7. Document retrieval from knowledge base.*

The process begins with an initial response based on the user's query (`request`), which serves as a foundation for generating a new `query` for targeted information retrieval. This is combined with user profile data (`user_info`) and the chosen retrieval channel (`channel_id`) to produce a comprehensive final response.

### Scripting Retrieval Response
This mode is designed for queries requiring detailed model information or specific actions, generating and executing Python scripts for tailored solutions.

    ![script-gen-exe-retrieval](./screenshots/script-gen-exe-retrieval.png)
    *Figure 8. Scripting retrieval for complex queries.*

It involves creating a `ModelZooInfoScript` schema with fields like `request`, `user info`, and `script`, where `script` is Python code for API interactions or data manipulation. The final response is formulated by integrating the script's output with the `request` and `user info`.

### API Call Response
This mode integrates user-provided APIs for data processing, currently implementing Cellpose[9] for image segmentation tasks on user-uploaded images.

    TODO: add an example screenshot.

The chatbot translates analysis requests into a `CellposeTask` schema, detailing the segmentation task per the `TaskChoice` schema. An LLM agent identifies image axes labels using a `LabeledImage` schema, ensuring compatibility with the segmentation process. Incompatible images trigger a `CellposeHelp` schema for guidance.

### Specialized Skill Response
This mode offers enriched responses for specific user needs, activating specialized modes like `Learning` and `Coding`.

    TODO: add an example screenshot.

The `Learning` mode provides educational responses enriched with key terms and concepts, while the `Coding` mode generates functional code snippets for specific tasks. These specialized responses are contrasted with the more general Direct Response mode and can be extended or chosen as a channel in the channel selection process.


## References

1. [FAISS](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)
2. [Bioimage.io](https://bioimage.io/docs/#/)
3. [Imjoy](https://imjoy.io/docs/#/)
4. [DeepImageJ](https://deepimagej.github.io/)
5. [ImageJ](https://imagej.net)
6. [bio.tools](https://bio.tools)
7. [scikit-image](https://scikit-image.org/docs/stable/)
8. [Function-Calling API](https://openai.com/blog/function-calling-and-other-api-updates)
9. [CellPose](https://www.cellpose.org)
