import os
from imjoy_rpc.hypha import login, connect_to_server
import asyncio
from imjoy_rpc.hypha import connect_to_server
import asyncio

results_css = """body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
}
.image-container {
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 5px;
    margin: 20px;
    text-align: center;
}
.image-container img {
    max-width: 100%;
    height: auto;
}
.download-button {
    background-color: #4CAF50;
    color: white;
    padding: 10px 20px;
    margin: 10px 0;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}
.download-button:hover {
    background-color: #45a049;
}"""

def npy_route_factory(npy_file_path):
    async def npy_route(event, context=None):
        with open(npy_file_path, "rb") as npy_file:
            npy_content = npy_file.read()
        return {
            "status": 200,
            "headers": {
                "Content-Type": "application/octet-stream",
                "Content-Disposition": f"attachment; filename={os.path.basename(npy_file_path)}"
            },
            "body": npy_content
        }
    return npy_route


def image_route_factory(image_file_path):
    async def image_route(event, context=None):
        with open(image_file_path, "rb") as img_file:
            img_content = img_file.read()
        return {
            "status": 200,
            "headers": {"Content-Type": "image/png"},
            "body": img_content
        }
    return image_route

def make_index_html(image_functions, npy_functions):
    index_html = f"""
        <!DOCTYPE html>
        <html>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <head>
            <title>Cellpose Segmentation Results</title>
            <style>
                {results_css}
            </style>
        </head>
        <body>"""
    for i, (img_nickname, _, desc) in enumerate(image_functions):
        npy_nickname, _, _ = npy_functions[i]
        index_html += f"""<div class="image-container">\n"""
        index_html += f"""<h2>{desc}</h2>\n"""
        index_html += f"""<img src="{img_nickname}" alt="Image">\n"""
        index_html += f"""<a href="{img_nickname}" download="{img_nickname}.png">\n"""
        index_html += f"""<button class="download-button">Download Image</button></a>\n"""
        index_html += f"""<a href="{npy_nickname}" download="{npy_nickname}.npy">\n"""
        index_html += f"""<button class="download-button">Download .npy</button></a>\n"""
        index_html += f"""</div>\n"""
    index_html += "</body></html>"
    async def index(event, context=None):
        return {
            "status": 200,
            "headers": {'Content-Type': 'text/html'},
            "body": index_html
        }
    return index

async def create_results_page(server_url, image_paths, image_headers, npy_paths):
    server = await connect_to_server({"server_url": server_url, "token": None, "method_timeout": 100})
    assert len(image_paths) == len(image_headers)
    image_functions = [(f"image_{i}", image_route_factory(image_path), desc) for i, (image_path, desc) in enumerate(zip(image_paths, image_headers))]
    npy_functions = [(f"npy_{i}", npy_route_factory(npy_path), desc) for i, (npy_path, desc) in enumerate(zip(npy_paths, image_headers))]
    index_page = make_index_html(image_functions, npy_functions)
    service_id = "chatbot-image-analysis"
    print(npy_functions)
    await server.register_service({
        "id": service_id,
        "type": "functions",
        "config": {
            "visibility": "public",
            "require_context": False
        },
        "index": index_page,
        **{nickname: route for nickname, route, _ in image_functions},
        **{nickname: route for nickname, route, _ in npy_functions},
    })
    server_url = server.config['public_base_url']
    user_url = f"{server_url}/{server.config['workspace']}/apps/{service_id}/index"
    print(f"Cellpose segmentation results are available at: {user_url}")
    return user_url

if __name__ == "__main__":
    server_url = "https://ai.imjoy.io"
    # server_url = "https://hypha.bioimage.io/"
    image_paths = ["tmp-output.png", "tmp-user-resized.png", "tmp-user-image.png"]
    image_headers = ["User results", "User resized image", "User image"]
    loop = asyncio.get_event_loop()
    loop.create_task(create_results_page(server_url, image_paths, image_headers))
    loop.run_forever()
