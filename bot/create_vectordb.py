import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)

# llm = OpenAI(temperature=0)
# Load the text file
# loader = TextLoader("/home/alalulu/workspace/chatbot_bmz/chatbot/text_files/CODE_OF_CONDUCT.txt")


# Read text_files folder to get all txt files including the ones in subfolders
def text_to_vectorstore(root_folder):
    txt_list = []
    for foldername, subfolders, filenames in os.walk(root_folder):
            for filename in filenames:
                if filename.endswith(".md"):
                    file_path = os.path.join(foldername, filename)
                    print(file_path)
                    loader = TextLoader(file_path)
                    documents = loader.load()

                    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                    texts=text_splitter.split_documents(documents)
                    txt_list.append(texts)
    return txt_list
root_folder = "/home/alalulu/workspace/chatbot_bmz/bioimage.io/docs"# "/home/alalulu/workspace/chatbot_bmz/chatbot/text_files"
txt_list = text_to_vectorstore(root_folder)
texts = [item for sublist in txt_list for item in sublist]
embeddings = OpenAIEmbeddings()
docs_store = Chroma.from_documents(
    texts, embeddings, collection_name="bioimage.io-docs"
)

vectorstore_info = VectorStoreInfo(
    name="BioImage Model Zoo",
    description="documentation from bioimage.io.",
    vectorstore=docs_store,
)
llm = OpenAI(temperature=0)
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True)

response = agent_executor.run(
    "Model contribution guidelines"
)
print(response)




# query = "can you show me the complete Model contribution guidelines?"
# docs = state_of_union_store.similarity_search(query)
# print(docs[0].page_content)
