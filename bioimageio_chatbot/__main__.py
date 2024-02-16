import sys
import argparse
import asyncio
import subprocess
import os
from bioimageio_chatbot.knowledge_base import load_knowledge_base

def start_server(args):
    if args.login_required:
        os.environ["BIOIMAGEIO_LOGIN_REQUIRED"] = "true"
    # get current file path so we can get the path of apps under the same directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    command = [
        sys.executable,
        "-m",
        "hypha.server",
        f"--host={args.host}",
        f"--port={args.port}",
        f"--public-base-url={args.public_base_url}",
        f"--static-mounts=/assistants:{current_dir}/apps/assistants/",
        "--startup-functions=bioimageio_chatbot.chatbot:register_chat_service"
    ]
    subprocess.run(command)

def connect_server(args):
    from bioimageio_chatbot.chatbot import connect_server
    if args.login_required:
        os.environ["BIOIMAGEIO_LOGIN_REQUIRED"] = "true"
    server_url = args.server_url
    loop = asyncio.get_event_loop()
    loop.create_task(connect_server(server_url))
    loop.run_forever()

def create_knowledge_base(args):
    from bioimageio_chatbot.knowledge_base import create_vector_knowledge_base
    create_vector_knowledge_base(args.output_dir)

def init(args):
    knowledge_base_path = os.environ.get("BIOIMAGEIO_KNOWLEDGE_BASE_PATH", "./bioimageio-knowledge-base")
    assert knowledge_base_path is not None, "Please set the BIOIMAGEIO_KNOWLEDGE_BASE_PATH environment variable to the path of the knowledge base."
    if not os.path.exists(knowledge_base_path):
        print(f"The knowledge base is not found at {knowledge_base_path}, will download it automatically.")
        os.makedirs(knowledge_base_path, exist_ok=True)
    docs_store_dict = load_knowledge_base(knowledge_base_path)
    
    print("Databases loaded in the knowledge base:")
    for key in docs_store_dict.keys():
        print(f" - {key}")

def main():
    parser = argparse.ArgumentParser(description="BioImage.IO Chatbot utility commands.")
    
    subparsers = parser.add_subparsers()

    # Init command
    parser_init = subparsers.add_parser("init")
    parser_init.set_defaults(func=init)

    # Start server command
    parser_start_server = subparsers.add_parser("start-server")
    parser_start_server.add_argument("--host", type=str, default="0.0.0.0")
    parser_start_server.add_argument("--port", type=int, default=9000)
    parser_start_server.add_argument("--public-base-url", type=str, default="")
    parser_start_server.add_argument("--login-required", action="store_true")
    parser_start_server.set_defaults(func=start_server)
    
    # Connect server command
    parser_connect_server = subparsers.add_parser("connect-server")
    parser_connect_server.add_argument("--server-url", default="https://ai.imjoy.io")
    parser_connect_server.add_argument("--login-required", action="store_true")
    parser_connect_server.set_defaults(func=connect_server)
    
    # Create knowledge base command
    parser_create_kb = subparsers.add_parser("create-knowledge-base")
    parser_create_kb.add_argument("--output-dir", default="./bioimageio-knowledge-base")
    parser_create_kb.set_defaults(func=create_knowledge_base)
    
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
        
if __name__ == '__main__':
    main()