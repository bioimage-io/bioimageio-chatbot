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

## Schema-Based Agent Design and Response Examples

The chatbot's ability to understand and respond to user queries is substantially improved by employing a schema-based agent design. Unlike traditional context-based models, our approach utilizes predefined schemas to guide the conversation and information retrieval process. 

The schema-based agent operates on the function-call LLM [8], and uses input and output schemas to generate text output. Within this implementation, we construct a customer service chatbot by defining a role class, as shown in Figure 4.

![role_create](./screenshots/role_create.png)
*Figure 4. Creation of a chatbot role class named ‘CustomerServiceRole’ by defining fields of the role class.*

### Information Retrieval

For straightforward queries, the chatbot generates concise and accurate answers directly from its pre-configured profile (see Figure 6).

    ![direct-response](./screenshots/direct-response.png)
    *Figure 6. Direct response for simple queries.*

In addition, there are two information retrieval modes:

1. **Response with Text-Based Retrieval**: For complex or specific questions, the chatbot extracts key queries from the user's question and retrieves information from relevant documentation (see Figure 7).

    ![retrieval-text](./screenshots/retrieval-text.png)
    *Figure 7. Text-based retrieval from the knowledge base.*

2. **Response with Script Generation and Execution**: For advanced queries requiring specific model details or actions, the chatbot dynamically generates and executes tailored scripts (see Figure 8).

    ![script-gen-exe-retrieval](./screenshots/script-gen-exe-retrieval.png)
    *Figure 8. Script generation and execution for advanced queries.*

## References

1. [FAISS](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)
2. [Bioimage.io](https://bioimage.io/docs/#/)
3. [Imjoy](https://imjoy.io/docs/#/)
4. [DeepImageJ](https://deepimagej.github.io/)
5. [ImageJ](https://imagej.net)
6. [bio.tools](https://bio.tools)
7. [scikit-image](https://scikit-image.org/docs/stable/)
8. [Function-Calling API](https://openai.com/blog/function-calling-and-other-api-updates)
