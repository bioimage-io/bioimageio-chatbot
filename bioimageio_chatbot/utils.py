
import requests
import yaml
import os
from tqdm import tqdm
from pydantic import BaseModel, Field
from typing import Callable, Optional
import typing
from inspect import signature

def get_manifest():
    # If no manifest is provided, download from the repo
    if not os.path.exists("./knowledge-base-manifest.yaml"):
        print("Downloading the knowledge base manifest...")
        response = requests.get("https://raw.githubusercontent.com/bioimage-io/bioimageio-chatbot/main/knowledge-base-manifest.yaml")
        assert response.status_code == 200
        with open("./knowledge-base-manifest.yaml", "wb") as f:
            f.write(response.content)
    
    return yaml.load(open("./knowledge-base-manifest.yaml", "r"), Loader=yaml.FullLoader)


def download_file(url, filename):
    response = requests.get(url, stream=True)
    file_size = int(response.headers.get('content-length', 0))

    # Initialize the progress bar
    progress = tqdm(response.iter_content(1024), f"Downloading {filename}", total=file_size, unit="B", unit_scale=True, unit_divisor=1024)
    
    with open(filename, 'wb') as f:
        for data in progress:
            # Update the progress bar
            progress.update(len(data))
            f.write(data)


def extract_schemas(function):
    sig = signature(function)
    positional_annotation = [
        p.annotation
        for p in sig.parameters.values()
        if p.kind == p.POSITIONAL_OR_KEYWORD
    ][0]
    output_schemas = (
        [sig.return_annotation]
        if not isinstance(sig.return_annotation, typing._UnionGenericAlias)
        else list(sig.return_annotation.__args__)
    )
    input_schemas = (
        [positional_annotation]
        if not isinstance(positional_annotation, typing._UnionGenericAlias)
        else list(positional_annotation.__args__)
    )
    return input_schemas, output_schemas

class ChatbotExtension(BaseModel):
    """A class that defines the interface for a user extension"""
    name: str = Field(..., description="The name of the extension")
    description: str = Field(..., description="A description of the extension")
    get_schema: Optional[Callable] = Field(None, description="A function that returns the schema for the extension")
    execute: Callable = Field(..., description="The extension's execution function")
    schema_class: Optional[BaseModel] = Field(None, description="The schema class for the extension")