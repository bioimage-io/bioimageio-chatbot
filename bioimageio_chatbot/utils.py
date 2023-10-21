
import requests
import yaml
import os

def get_manifest():
    # If no manifest is provided, download from the repo
    if not os.path.exists("./knowledge-base-manifest.yaml"):
        print("Downloading the knowledge base manifest...")
        response = requests.get("https://raw.githubusercontent.com/bioimage-io/bioimageio-chatbot/main/knowledge-base-manifest.yaml")
        assert response.status_code == 200
        with open("./knowledge-base-manifest.yaml", "wb") as f:
            f.write(response.content)
    
    return yaml.load(open("./knowledge-base-manifest.yaml", "r"), Loader=yaml.FullLoader)
