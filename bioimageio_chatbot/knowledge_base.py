import os
import requests
import zipfile
import shutil
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
import json
import pickle
from bioimageio_chatbot.utils import get_manifest, download_file
from bioimageio_chatbot.parent_retriever import build_parent_retriever, load_parent_retriever

KNOWLEDGE_BASE_URL = os.environ.get("BIOIMAGEIO_KNOWLEDGE_BASE_URL", "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimageio-knowledge-base")


def load_retriever(db_path, collection_name):
    # Each collection has two files [collection_name].faiss and [collection_name].pkl
    # Check if it exists, otherwise, download from {KNOWLEDGE_BASE_URL}/[collection].faiss
    if not os.path.exists(os.path.join(db_path, f"{collection_name}.faiss")):
        print(f"Downloading {collection_name}.faiss from {KNOWLEDGE_BASE_URL}/{collection_name}.faiss")
        download_file(f"{KNOWLEDGE_BASE_URL}/{collection_name}.faiss", os.path.join(db_path, f"{collection_name}.faiss"))
    
    if not os.path.exists(os.path.join(db_path, f"{collection_name}.pkl")):
        print(f"Downloading {collection_name}.pkl from {KNOWLEDGE_BASE_URL}/{collection_name}.pkl")
        download_file(f"{KNOWLEDGE_BASE_URL}/{collection_name}.pkl", os.path.join(db_path, f"{collection_name}.pkl"))

    # Load from vector store
    embeddings = OpenAIEmbeddings()
    # docs_store = FAISS.load_local(index_name=collection_name, folder_path=db_path, embeddings=embeddings)
    retriever = load_parent_retriever(output_dir=db_path, index_name=collection_name, embeddings=embeddings)
    return retriever


def load_knowledge_base(db_path):
    collections = get_manifest()['collections']
    retriever_dict = {}
    
    for collection in collections:
        channel_id = collection['id']
        try:
            retriever = load_retriever(db_path, channel_id)
            
            # length = len(docs_store.docstore._dict.keys())
            # assert length > 0, f"Please make sure the docs store {channel_id} is not empty."
            # print(f"Loaded {length} documents from {channel_id}")
            retriever_dict[channel_id] = retriever
        except Exception as e:
            print(f"Failed to load docs store for {channel_id}. Error: {e}")

    if len(retriever_dict) == 0:
        raise Exception("No docs store is loaded, please make sure the docs store is not empty.")

    return retriever_dict

def extract_biotools_information(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    extracted_info = []
    data['url'] = f"https://bio.tools/{data['name']}"
    # Extracting required information
    if 'name' in data:
        extracted_info.append(f"Name: {data['name']}")
    if 'description' in data:
        extracted_info.append(f"Description: {data['description']}")
    
    if 'toolType' in data:
        extracted_info.append(f"Tags: {', '.join(data['toolType'])}")
        
    if 'topic' in data:
        topics = [item['term'] for item in data['topic']]
        extracted_info.append(f"Topics: {', '.join(topics)}")
    
    if 'publication' in data:
        for pub in data['publication']:
            if 'metadata' in pub and 'authors' in pub['metadata']:
                authors = [author['name'] for author in pub['metadata']['authors']]
                extracted_info.append(f"Publication Authors: {', '.join(authors)}")
    # Write extracted information to text file
    return "\n".join(extracted_info), data

# Read text_files folder to get all txt files including the ones in subfolders
def parse_docs(root_folder, md_separator=None, pdf_separator=None, chunk_size=1000, chunk_overlap=10):
    chunk_list = []
    for foldername, _, filenames in os.walk(root_folder):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                if filename.endswith(".md"):
                    print(f"Reading {file_path}...")
                    documents = TextLoader(file_path).load()
                    text_splitter = CharacterTextSplitter(separator=md_separator or "\n#", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    chunks =text_splitter.split_documents(documents)
                elif filename.endswith(".pdf"):
                    print(f"Reading {file_path}...")
                    documents = PyPDFLoader(file_path).load()
                    text_splitter = RecursiveCharacterTextSplitter(separators=pdf_separator or ["\n\n", "\n", " ", ""], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    chunks = text_splitter.split_documents(documents)    
                elif filename.endswith(".txt"):
                    print(f"Reading {file_path}...")
                    documents = TextLoader(file_path).load()
                    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    chunks = text_splitter.split_documents(documents)
                elif filename.endswith(".biotools.json"):
                    # convert json to yaml
                    print(f"Reading {file_path}...")
                    content, metadata = extract_biotools_information(file_path)
                    chunks = [Document(page_content=content, metadata=metadata)]         
                else:
                    print(f"Skipping {file_path}")
                    continue
                chunk_list.extend(chunks)
                    
    return chunk_list

def download_docs(root_dir, url):
    os.makedirs(root_dir, exist_ok=True)
    # extract filename from url, remove query string
    filename = url.split("/")[-1].split("?")[0]
    # target directory is ./repos
    target_directory = os.path.join(root_dir)
    # if the target directory exists, remove it anyway and create a new one
    if os.path.exists(target_directory):
        shutil.rmtree(target_directory)
    os.mkdir(target_directory)
    if filename.endswith(".zip"):
        # Define the file and folder names
        zip_file_path = os.path.join(target_directory, filename)
        print(f"Downloading {url} to {zip_file_path}")
        # Download the ZIP file
        download_file(url, zip_file_path)

        result_folder = os.path.join(target_directory, filename + "-unzipped")
        # Unzip the downloaded file 
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(result_folder)
        
        # Clean up - remove the downloaded ZIP file
        os.remove(zip_file_path)
        print(f"Downloaded and unzipped {url} to {result_folder}")
    elif filename.endswith(".pdf"):
        result_folder = os.path.join(target_directory, ".".join(filename.split(".")[:-1]))
        os.makedirs(result_folder, exist_ok=True)
        print(f"Downloading {url} to {result_folder}")
        pdf_file_path = os.path.join(result_folder, filename)
        download_file(url, pdf_file_path)
        print(f"Downloaded {url} to {result_folder}")
        
    else:
        raise Exception("Unsupported file format")
    
    if len(os.listdir(result_folder)) == 0:
        raise Exception("Downloaded folder is empty")
    elif len(os.listdir(result_folder)) == 1:
        # strip the folder name of the unzipped repo
        r_dir = os.path.join(result_folder, os.listdir(result_folder)[0])
        if os.path.isdir(r_dir):
            return r_dir
    # get the folder name of the unzipped repo
    return result_folder


def create_vector_knowledge_base(output_dir=None, collections=None):
    """Create a vector knowledge base from the downloaded documents"""
    if output_dir is None:
        output_dir = os.environ.get("BIOIMAGEIO_KNOWLEDGE_BASE_PATH", "./bioimageio-knowledge-base")
    os.makedirs(output_dir, exist_ok=True)
    manifest = get_manifest()
    if not collections:
        collections = manifest['collections']
        
    embeddings = OpenAIEmbeddings()
    for collection in collections:
        url = collection['source']
        cached_docs_file = os.path.join(output_dir, collection['id'] + "-docs.pickle")
        if os.path.exists(cached_docs_file):
            with open(cached_docs_file, "rb") as f:
                documents = pickle.load(f)
        else:    
            docs_dir = download_docs("./data", url)
            documents = parse_docs(os.path.join(docs_dir, collection.get('directory', '')),md_separator=collection.get('md_separator', None), pdf_separator=collection.get('pdf_separator', None), chunk_size=collection.get('chunk_size', 1000), chunk_overlap=collection.get('chunk_overlap', 10))
        if len(documents) > 10000:
            print(f"Waring: {len(documents)} documents found in {url}.")
        # save the vector db to output_dir
        print(f"Creating embeddings (#documents={len(documents)}))")
        child_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". "], chunk_size=manifest.get('child_chunk_size', 250), chunk_overlap=collection.get('child_chunk_overlap', 0))
        build_parent_retriever(documents, index_name=collection['id'], embeddings=embeddings, child_splitter=child_splitter, save_dir=output_dir)



        # # Choose an appropriate batch size
        # batch_size = 1000 

        # # Initialize an empty list to store all the batch_embedding_pairs
        # all_embedding_pairs = []
        # all_metadata = []

        # total_length = len(documents)

        # # Loop over your documents in batches
        # for batch_start in range(0, total_length, batch_size):
        #     batch_end = min(batch_start + batch_size, total_length)
        #     batch_texts = documents[batch_start:batch_end]

        #     # Generate embeddings for the batch of texts
        #     batch_embeddings = embeddings.embed_documents([t.page_content for t in batch_texts])
        #     batch_embedding_pairs = zip([t.page_content for t in batch_texts], batch_embeddings)

        #     # Append the batch_embedding_pairs to the all_embedding_pairs list
        #     all_embedding_pairs.extend(batch_embedding_pairs)
        #     all_metadata.extend([t.metadata for t in batch_texts])

        #     print(f"Processed {batch_end}/{total_length} documents")

        # # Create the FAISS index from all the embeddings
        # vectordb = FAISS.from_embeddings(all_embedding_pairs, embeddings, metadatas=all_metadata)
        # print("Saving the vector database...")
        # vectordb.save_local(output_dir, index_name=collection['id'])
        print("Created a vector database from the downloaded documents.")

if __name__ == "__main__":
    create_vector_knowledge_base()