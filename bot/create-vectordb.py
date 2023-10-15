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

# Read text_files folder to get all txt files including the ones in subfolders
def md_to_vectorstore(root_folder):
    txt_list = []
    for foldername, subfolders, filenames in os.walk(root_folder):
            for filename in filenames:
                if filename.endswith(".md"):
                    file_path = os.path.join(foldername, filename)
                    print(file_path)
                    loader = TextLoader(file_path)
                    documents = loader.load()

                    text_splitter = CharacterTextSplitter(separator="\n## ", chunk_size=1000, chunk_overlap=50)
                    texts=text_splitter.split_documents(documents)
                    txt_list.append(texts)
    return txt_list
# root_folder = "/home/alalulu/workspace/chatbot_bmz/bioimage.io/docs"# "/home/alalulu/workspace/chatbot_bmz/chatbot/text_files"
# txt_list = md_to_vectorstore(root_folder)
# texts = [item for sublist in txt_list for item in sublist]

# load from vectorstore
embeddings = OpenAIEmbeddings()
output_dir="docs/vectordb"
docs_store = Chroma(collection_name="bioimage.io-docs", persist_directory=output_dir, embedding_function=embeddings)



question = "what are Model contribution guidelines?"
docs = docs_store.similarity_search(question)
raw_docs = []
for doc in docs:
    # combine all the docs into one string
    raw_docs.append("```markdown\n" + doc.page_content + "\n```")
raw_docs = "\n".join(raw_docs)



