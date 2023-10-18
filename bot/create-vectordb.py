import os
import requests
import zipfile
import shutil
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
from langchain.llms import OpenAI

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

def download_bioimageio_docs():
    # URL of the ZIP file
    url = "https://github.com/bioimage-io/bioimage.io/archive/refs/heads/main.zip"

    # Define the file and folder names
    zip_file_name = "main.zip"

    # target directory is ./repos
    target_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")

    # if the target directory exists, remove it anyway and create a new one
    if os.path.exists(target_directory):
        shutil.rmtree(target_directory)
    os.mkdir(target_directory)

    # Download the ZIP file
    response = requests.get(url)
    with open(os.path.join(target_directory, zip_file_name), "wb") as zip_file:
        zip_file.write(response.content)

    # Unzip the downloaded file
    with zipfile.ZipFile(os.path.join(target_directory, zip_file_name), "r") as zip_ref:
        zip_ref.extractall(target_directory)

    # Clean up - remove the downloaded ZIP file
    os.remove(os.path.join(target_directory, zip_file_name))

    print("Downloaded, unzipped bioimage-io repo.")


def create_vecdb():
    target_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
    # check if the target directory exists
    if not os.path.exists(target_directory):
        raise Exception("The target directory doesn't exist. Please run `download_bioimageio_docs()` first.")
    
    docs_directory = os.path.join(target_directory, "bioimage.io-main/docs")

    txt_list = md_to_vectorstore(docs_directory)
    texts = [item for sublist in txt_list for item in sublist]

    embeddings = OpenAIEmbeddings()
    # db = Chroma.from_documents(texts, embeddings)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../docs")
    # check if the output directory exists, if exists, remove it and create a new one
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    # save the vector db to output_dir
    db = Chroma.from_documents(texts, embeddings, collection_name="bioimage.io-docs", persist_directory=output_dir)
    print("Created a vector database from the downloaded documents.")

if __name__ == "__main__":
    download_bioimageio_docs()
    create_vecdb()

    
 