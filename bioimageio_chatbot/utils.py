
import requests
import yaml
import os
from tqdm import tqdm

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
