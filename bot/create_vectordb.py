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
from simpleaichat import AIChat
# llm = OpenAI(temperature=0)
# Load the text file
# loader = TextLoader("/home/alalulu/workspace/chatbot_bmz/chatbot/text_files/CODE_OF_CONDUCT.txt")


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
root_folder = "/home/alalulu/workspace/chatbot_bmz/bioimage.io/docs"# "/home/alalulu/workspace/chatbot_bmz/chatbot/text_files"
txt_list = md_to_vectorstore(root_folder)
texts = [item for sublist in txt_list for item in sublist]
embeddings = OpenAIEmbeddings()
output_dir="docs/vectordb"
# docs_store = Chroma.from_documents(
#     texts, embeddings, persist_directory=output_dir, collection_name="bioimage.io-docs"
# )



PREFIX = """Your name is BMZ. You are an chatbot designed to answer questions about sets of documents. 
If you are given a set of raw documents retrieved from a search engine in the `Context`, you will answer the question based on the `Context`.
If the question does not seem to be relavant to the `Context`, just return "I don't know" as the answer.
"""

ai = AIChat(system=PREFIX)

question = "what are Model contribution guidelines?"
docs = docs_store.similarity_search(question)
raw_docs = []
for doc in docs:
    # combine all the docs into one string
    raw_docs.append("```markdown\n" + doc.page_content + "\n```")
raw_docs = "\n".join(raw_docs)

prompt = f"##Context\n{raw_docs}\n##Question\n{question}\nNow, please anwser the user's question based the context."
response = ai(prompt)
print(response)

