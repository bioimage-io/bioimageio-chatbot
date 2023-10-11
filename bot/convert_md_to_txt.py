import markdown2
import re

# Define a function to extract links from a Markdown file
def extract_links_from_md(md_file):
    # Read the Markdown content from the file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Parse the Markdown content into HTML
    html_content = markdown2.markdown(md_content)

    # Use regular expressions to find links to websites and figures
    website_links = re.findall(r'href=["\'](https?://[^"\']+)["\']', html_content)
    figure_links = re.findall(r'src=["\']([^"\']+)["\']', html_content)

    return website_links, figure_links

# Define the input Markdown file and output text file
input_md_file = 'input.md'
output_txt_file = 'links.txt'

# Extract links
website_links, figure_links = extract_links_from_md(input_md_file)

# Save the links to a text file
with open(output_txt_file, 'w', encoding='utf-8') as f:
    f.write("Website Links:\n")
    for link in website_links:
        f.write(link + '\n')

    f.write("\nFigure Links:\n")
    for link in figure_links:
        f.write(link + '\n')

print(f"Links extracted and saved to {output_txt_file}")


if __name__ == "__main__":
    input_file = "/home/alalulu/workspace/chatbot_bmz/chatbot/bot/how_to_join.md"  # Replace with the path to your Markdown file
    output_file = "output.txt"  # Replace with the desired output file path

    markdown_to_text(input_file, output_file)
