import sys
import argparse
import asyncio
import subprocess
import os

def start_server(args):
    print(sys.executable)
    command = [
        sys.executable,
        "-m",
        "hypha.server",
        f"--host={args.host}",
        f"--port={args.port}",
        "--startup-functions=bioimageio_chatbot.chatbot:register_chat_service"
    ]
    subprocess.run(command)

def connect_server(args):
    from bioimageio_chatbot.chatbot import connect_server
    server_url = args.server_url
    loop = asyncio.get_event_loop()
    loop.create_task(connect_server(server_url))
    loop.run_forever()

def create_knowledge_base(args):
    from bioimageio_chatbot.knowledge_base import create_vector_knowledge_base
    create_vector_knowledge_base(args.output_dir)


def main():
    parser = argparse.ArgumentParser(description="BioImage.IO Chatbot utility commands.")
    
    subparsers = parser.add_subparsers()
    
    # Start server command
    parser_start_server = subparsers.add_parser("start-server")
    parser_start_server.add_argument("--host", type=str, default="0.0.0.0")
    parser_start_server.add_argument("--port", type=int, default=9000)
    parser_start_server.set_defaults(func=start_server)
    
    # Connect server command
    parser_connect_server = subparsers.add_parser("connect-server")
    parser_connect_server.add_argument("--server-url", default="https://ai.imjoy.io")
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