# Examples

## ChatBot interface

Following installation guidelines from the [README](/README.md), the chat interface looks like Figure 1.

![BioImage.IO-ChatBot](./screenshots/chat-interface.png)
*Figure 1. The chat interface of BioImage.IO ChatBot.*

User can input their profile as shown in Figure 2. 
![user-profile](./screenshots/user-profile.png).
*Figure 2. Users can input their profile by clicking `Edit Profile` to personalized responses and `Save` to save their profile for future conversations.*

As for today, our knowledge bases for the chatbot are drawn from documentation spanning below pivotal communities: bioimage.io [2], Imjoy [3], deepimageJ [4], ImageJ [5], bio.tools [6] and scikit-image [7]. A key feature of the implementation is the user's ability to specify a channel from which they prefer to retrieve information as shwon in Figure 3. If the user designates a channel, the chatbot seamlessly sources information from that specific community. However, in cases where the channel is not specified, the chatbot utilizes an intelligent selection process guided by the schema-based agent to determine the most relevant channel based on the user's question. We will elaborate on the schema-based agent in the following, which plays a central role in this seamless and efficient retrieval process.

![channels](./screenshots/channels.png)
*Figure 3. Users can select a specific channel from ‘Knowledge Base Channel’ to personalize the conversation.*

The construction of the knowledge base is both efficient and collaborative. We begin by downloading the documentation from a given URL, which could be a repository, a PDF, or any other form of documentation. To better organize and facilitate retrieval, we employ a specified splitter of regular expression patterns (regex splitter) to automatically segment the documentation into more manageable chunks for efficient and accurate retrieval. These segmented documentation chunks are then embedded and stored as vectors within the vector database. We use the FAISS [1] based vector database in the implementation.


## Schema-Based Agent Design adn Response Examples

Our chatbot's efficacy in understanding and responding to user queries is significantly enhanced through the innovative approach of schema-based agent design. Instead of relying solely on pure-context prompts, which often fall short when extracting specific information from user queries or understanding complex requirements, we employ predefined schemas to a schema-based agent to guide the conversation and information retrieval process. Schemas are data structures that represent various kinds of knowledge. In agent design, schemas often determine how an agent perceives its environment, makes decisions, and acts upon those decisions. They can represent anything from simple tasks to complex actions or sequences.

The schema-based agent is built upon the function-call LLM [8], and it utilizes input and output schema to generate text output. Specifically, by passing a union of pydantic forms (schemas) for the output, the schema agent will decide which schema to use for generating the output based on the given input context. Within the implementation, we create the customer service chatbot by inputting the fields of a role class as shown in Figure 4. 
![role_create](./screenshots/role_create.png)
*Figure 4. Creating a chatbot role class named ‘CustomerServiceRole’ by inputting the fields of a role class.*

When responding to the user, we have designed three distinct response modes to optimize the chatbot's service: Direct Response, Response from Documentation Retrieval, and Script Generation and Execution, as detailed in the following context. To determine which response mode is best suited for a given query, we integrate the user's original question with the conversation history and the user's profile into a single input. This combined input, along with the three schemas, is passed to the schema agent, enabling it to decide the appropriate response mode for answering the question as shwon in Figure 5. 
![respond_to_user](./screenshots/respond_to_user.png)
*Figure 5. Integrate the user's original question with the conversation history and the user's profile into a single input `inputs`. The inputs along with the three schemas is passed to the schema agent via calling `role.aask()` enabling the agent to reformulate the questions and decide the appropriate response mode for answering the question.*

This step is not only helpful for selecting the suitable response schema for the schema agent to provide more efficient and accurate answers, but it is also necessary to reorganize the original question into a clear and concise form for better understanding and more efficient retrieval in the following.  

Now, let's explore how these response modes operate to offer tailored and effective interactions with users. The Three Response Modes are:

**Direct Response**: In the case of straightforward questions, where the user seeks basic information about the BioImage Model Zoo or related topics, our chatbot provides direct responses. For instance, if a user inquires, "What is BioImage Model Zoo?" The chatbot can readily generate a concise and accurate answer, drawing from its pre-configured profile (see Figure 6).
![direct-response](./screenshots/direct-response.png)
*Figure 6. Direct response for simple and easy questions.*

**Response With Text-based Retrieval**: When users ask more complex or specific questions about bioimage analysis information like bioimage.io, model zoo, or Imjoy, the chatbot responds using information retrieved from the corresponding database. The chatbot first extracts key queries from the user's question, and then utilizes these key queries to retrieve relevant information from the BioImage Model Zoo documentation repository. By incorporating schema-based agent, the chatbot can better understand and analyze user queries, significantly improving the accuracy of document retrieval. For example, if a user asks, "How can I upload a model to the BioImage Model Zoo?" The chatbot can effectively identify the key query as "model upload," retrieve pertinent documents and generate a comprehensive response (see Figure 7).

![retrieval-text](./screenshots/retrieval-text.png)
*Figure 7. Response by retrieving text information from the knowledge base.*

**Response With Script Generation and Execution**: Some user queries may require a more intricate response. In cases where users seek specific model details or actions, the chatbot can dynamically create scripts (like python) tailored to the user's needs. It then executes these scripts using the available model resource files. For example, if a user inquires, "What models in the Model Zoo can perform segmentation tasks?," the chatbot can automatically generate and execute a script to identify and list models suitable for image segmentation (see Figure 8).
![script-gen-exe-retrieval](./screenshots/script-gen-exe-retrieval.png)
*Figure 8. Response by generating code script and executing the script using the available model resource item to retrieve information for responding.*

This schema-based agent design not only enables the chatbot to differentiate and adapt to diverse user queries but also significantly enhances the overall user experience. It ensures that users receive responses that are not only relevant but also presented in a structured and understandable manner.

## Customization Examples

Customization is a crucial element in our chatbot's design, especially in the dynamic field of bioimage analysis, which attracts users from diverse backgrounds such as developers, researchers, and biologists. We've harnessed the flexibility of the schema-agent to enable users to provide information about their background. For instance, if a user has a background in bioimage analysis, this information is seamlessly integrated into their interactions with the chatbot (see see Figure 9). 
![customization2_developer](./screenshots/customization2_developer.png)
*Figure 9. Customerization for user with bioimage analysis background.*

On the other hand, if a user lacks a bioimage analysis background but submits the same query, the chatbot utilizes the user's profile information to respond in a way that ensures better understanding and a more tailored experience (see Figure 10).
![customization2_developer](./screenshots/customization2_developer.png)
*Figure 10. Customerization for user without any bioimage analysis background.* 

 The customization also offers users the option to specify a channel from which they prefer to retrieve information. If the user designates a channel, the chatbot seamlessly sources information from that specific community (see Figure 11). 
![customization_biotool](./screenshots/customization_biotool.png)
*Figure 11. Customerization when selecting specific channel ‘bio.tools’.*


## Reference

1. [FAISS](https://github.com/bioimage-io/bioimageio-chatbot)
2. [Bioimage.io](https://bioimage.io/docs/#/)
3. Imjoy, Ouyang, Ouyang, Wei, Florian Mueller, Martin Hjelmare, Emma Lundberg, and Christophe Zimmer. "ImJoy: an open-source computational platform for the deep learning era." Nature methods 16, no. 12 (2019): 1199-1200. [Imjoy Documentation](https://imjoy.io/docs/#/)
4. deepimageJ, Gómez-de-Mariscal, Estibaliz, Carlos García-López-de-Haro, Wei Ouyang, Laurene Donati, Emma Lundberg, Michael Unser, Arrate Munoz-Barrutia, and Daniel Sage. "DeepImageJ: A user-friendly environment to run deep learning models in ImageJ." Nature Methods 18, no. 10 (2021): 1192-1195. [DeepImageJ Documentation](https://deepimagej.github.io/)
5. [ImageJ](https://imagej.net)
6. [bio.tools](https://bio.tools)
7. [scikit-image](https://scikit-image.org/docs/stable/)
8. [Function calling API](https://openai.com/blog/function-calling-and-other-api-updates)
