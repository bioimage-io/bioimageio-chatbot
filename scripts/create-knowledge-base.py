import os
import requests
import zipfile
import shutil
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings

TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Read text_files folder to get all txt files including the ones in subfolders
def parse_docs(root_folder, md_separator=None, pdf_separator=None, chunk_size=1000, chunk_overlap=10):
    chunk_list = []
    for foldername, _, filenames in os.walk(root_folder):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                if filename.endswith(".md"):
                    print(f"Reading {file_path}...")
                    documents = TextLoader(file_path).load()
                    text_splitter = CharacterTextSplitter(separator=md_separator or "\n## ", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    chunks =text_splitter.split_documents(documents)
                elif filename.endswith(".pdf"):
                    print(f"Reading {file_path}...")
                    documents = PyPDFLoader(file_path).load()
                    text_splitter = RecursiveCharacterTextSplitter(separators=pdf_separator or ["\n\n", "\n", " ", ""], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    chunks = text_splitter.split_documents(documents)                
                else:
                    print(f"Skipping {file_path}")
                    continue
                chunk_list.extend(chunks)
                    
    return chunk_list

def download_docs(url):
    # extract filename from url, remove query string
    filename = url.split("/")[-1].split("?")[0]
    # target directory is ./repos
    target_directory = os.path.join(TEMP_DIR, os.path.basename(filename))
    # if the target directory exists, remove it anyway and create a new one
    if os.path.exists(target_directory):
        shutil.rmtree(target_directory)
    os.mkdir(target_directory)
    if filename.endswith(".zip"):
        # Define the file and folder names
        zip_file_name = "main.zip"
        

        zip_file_path = os.path.join(target_directory, zip_file_name)
        # Download the ZIP file
        response = requests.get(url)
        with open(zip_file_path, "wb") as zip_file:
            zip_file.write(response.content)

        # Unzip the downloaded file 
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(target_directory)
        # Clean up - remove the downloaded ZIP file
        os.remove(zip_file_path)
        print(f"Downloaded and unzipped {url} to {target_directory}")
    elif filename.endswith(".pdf"):
        response = requests.get(url)
        pdf_file_path = os.path.join(target_directory, filename)
        with open(pdf_file_path, "wb") as pdf_file:
            pdf_file.write(response.content)
        print(f"Downloaded {url} to {target_directory}")
    # get the folder name of the unzipped repo
    return target_directory
    

def create_vector_knowledge_base(collections, output_dir):
    embeddings = OpenAIEmbeddings()
    for collection in collections:
        url = collection['url']
        docs_dir = download_docs(url)
        documents = parse_docs(docs_dir)
        # save the vector db to output_dir
        print(f"Creating embeddings (#documents={len(documents)}))")
        vectordb = FAISS.from_documents(documents, embeddings)
        vectordb.save_local(output_dir, index_name=collection['id'])
        print("Created a vector database from the downloaded documents.")


if __name__ == "__main__":
    import yaml
    manifest = yaml.load(open("./manifest.yaml", "r"), Loader=yaml.FullLoader)
    create_vector_knowledge_base(manifest['collections'], 'docs/knowledge-base')

    vectordb = FAISS.load_local(folder_path="docs/knowledge-base", index_name="scikit-image", embeddings=OpenAIEmbeddings())
    retriever = vectordb.as_retriever(score_threshold=0.4)
    items = retriever.get_relevant_documents("scikit-image release", verbose=True)
    print(items)
 