import asyncio
from imjoy_rpc.hypha import login, connect_to_server
from langchain.llms import OpenAI
from dotenv import load_dotenv

# Load .env file so we have OPENAI_API_KEY
load_dotenv()

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)

llm = OpenAI(temperature=0)
from langchain.document_loaders import TextLoader

# loader = TextLoader("/home/alalulu/workspace/hypha-bot/scripts/imageJ_doc.txt")
loader = TextLoader("/home/alalulu/workspace/chatbot_bmz/chatbot/text_files/CODE_OF_CONDUCT.txt")  

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
state_of_union_store = Chroma.from_documents(
    texts, embeddings, collection_name="state-of-union"
)


vectorstore_info = VectorStoreInfo(
    name="imageJ_database",
    description="documentation of imageJ",
    vectorstore=state_of_union_store,
)
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True)

# response = agent_executor.run(
#     "what is imagej"
# )

# print(response)
async def start_server(server_url):
    token = await login({"server_url": server_url})
    server = await connect_to_server({"server_url": server_url, "token": token})
    # llm = OpenAI(temperature=0.9)
    
    async def chat(text, api, context=None):
        response = agent_executor.run(
            text
        )
        print(f"\nUser: {text}\nBot: {response}")
        return response

    await server.register_service({
        "name": "Hypha Bot",
        "id": "hypha-bot",
        "config": {
            "visibility": "public",
            "require_context": True
        },
        "chat": chat
    })

    # print(f"hello world service regisered at workspace: {server.config.workspace}")
    # print(f"Test it with the http proxy: {server_url}/{server.config.workspace}/services/hello-world/hello?name=John")

if __name__ == "__main__":
    server_url = "https://ai.imjoy.io"
    loop = asyncio.get_event_loop()
    loop.create_task(start_server(server_url))
    loop.run_forever()