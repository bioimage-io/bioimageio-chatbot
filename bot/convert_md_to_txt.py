import os
import markdown2

# Get the current working directory
current_directory = os.getcwd()

# Define the directory where your cloned repository is located
repo_directory= r"C:\Users\CaterinaFusterBarcel\Documents\GitHub\bioimage.io\docs"
# repo_directory = "/home/alalulu/workspace/chatbot_bmz/bioimage.io/docs"

# Get the name of the repository
repo_name = repo_directory.split("GitHub")[1][1:]
repo_name = repo_name.replace("\\", "-")
repo_name = repo_name.replace("/", "-")

# Create a directory for text files if it doesn't exist in the current directory
txt_directory = os.path.join(current_directory, repo_name)
os.makedirs(txt_directory, exist_ok=True)

for root, dirs, files in os.walk(repo_directory):
    for file in files:
        if file.endswith(".md"):
            # Read the Markdown file
            with open(os.path.join(root, file), 'r', encoding='utf-8') as md_file:
                markdown_content = md_file.read()

            # Convert Markdown to HTML
            html_content = markdown2.markdown(markdown_content)

            # Save HTML to a text file in the current directory
            relative_path = os.path.relpath(root, repo_directory)
            txt_file_dir = os.path.join(txt_directory, relative_path)
            os.makedirs(txt_file_dir, exist_ok=True)

            txt_file = os.path.splitext(file)[0] + ".txt"
            with open(os.path.join(txt_file_dir, txt_file), 'w', encoding='utf-8') as txt_file:
                txt_file.write(html_content)

print("Markdown to text conversion completed. Text files are saved in the current directory.")

