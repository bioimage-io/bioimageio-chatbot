# Decoding Conversational Assistants: BioImage.IO Chatbot and ChatGPT

## Introduction 
The document aims to compare and contrast the BioImage.IO Chatbot with ChatGPT, addressing their differences and similarities. The community has expressed curiosity regarding how these two conversational assistants differ, particularly given their distinct purposes. ChatGPT is a generalist tool designed to assist and provide information based on user inputs, regardless of their scope. In contrast, the BioImage.IO Chatbot specializes in bioimage analysis, particularly in relation to the BioImage Model Zoo and other software within its knowledge domain. 

## Feature Comparison
The subsequent sections will analyze and compare specific features of both chatbots, highlighting their differences and similarities in each area. 

### Context Awareness
ChatGPT 3.5 (free version) does not possess full context awareness. Each query or input is processed as an isolated instance, even within the same conversation. As a result, it lacks a continuous understanding of the broader dialogue or the capacity to recall past interactions. Conversely, the BioImage.IO Chatbot exhibits context awareness, remembering past interactions and leveraging this information to enhance user responses. 


### Personalization
ChatGPT 3.5 lacks the capacity to personalize responses for individual users. On the other hand, the BioImage.IO Chatbot is designed to customize responses based on optional user-provided information, such as background and specific needs. This approach allows the BioImage.IO Chatbot to offer varied feedback tailored to the user's unique profile. 

### Code generation

### User Experience

## Knowledge Retrieval Evaluation
An evaluation method has been developed to compare the knowledge retrieval capabilities of the BioImage.IO Chatbot with ChatGPT versions 3.5 and 4.0. This method, outlined in the provided repository [insert link here](), involves creating a Ground Truth using GPT-3.5-turbo. In this step, a question is answered with the documentation serving as context. The same question is then posed to ChatGPT 3.5, ChatGPT 4.0, and the BioImage.IO Chatbot. For each response from these chatbots, a Similarity Check is conducted against the Ground Truth to determine a Similarity Score. The process is depicted in the figure below:

![Similarity Check - provisional](./screenshots/similarity-check.jpg)

To illustrate this evaluation method, specific cases will be detailed in the subsequent queries:

### Who are the main contributors to ilastik?
#### Answers

* **ChatGPT 3.5, *Similarity score: 2.5:*** The main contributors to ilastik are a group of researchers and developers from the ilastik team at the Biomedical Imaging Group at the Center for Systems Biology at the Max Planck Institute of Molecular Cell Biology and Genetics in Dresden, Germany. The team includes experts in computer vision, machine learning, and image processing, who work together to develop and maintain the ilastik software. 
* **ChatGPT 4.0, *Similarity score: 4.5:*** The main contributors to Ilastik, a popular software for interactive image analysis, are primarily from the scientific research community. Key contributors include: Anna Kreshuk - She has been instrumental in the development of Ilastik and is known for her work in machine learning and image analysis. Fred A. Hamprecht - A professor at Heidelberg University, Hamprecht's lab has played a significant role in the development of Ilastik, focusing on computer vision and machine learning. Carsten Haubold - Involved in the development, focusing on the integration of machine learning methods.Teams from European Molecular Biology Laboratory (EMBL) and Heidelberg University: Collaborative efforts between these institutions have been central to Ilastik's development. These contributors, along with many others in the scientific community, have made significant developments to the software, contributing to its capabilities in image segmentation, classification, and analysis.
* **BioImage.IO Chatbot, *Similarity score: 1:*** The main contributors to ilastik are not explicitly listed in the provided documentation excerpts. However, ilastik is presented as an open-source tool under the BSD license, which invites collaboration and contributions from developers in the field of biomedical image processing. For more detailed information about the contributors, you may want to check the [ilastik homepage](https://www.ilastik.org/documentation/) or contact the community via the [mailing list](mailto:ilastik-user@ilastik.org).



#### Evaluation
This section demonstrates how the three conversational assistants respond to a complex question. In this instance, ChatGPT 3.5 not only failed to answer the question correctly but also fabricated a response. ChatGPT 4.0 managed to answer the question accurately but was unable to provide a link to the relevant documentation. On the other hand, the BioImage.IO Chatbot, while unable to answer the question directly, was aware of its limitations. As a result, despite not having the exact answer, it successfully provided a link to the documentation. 

## Conclusions
