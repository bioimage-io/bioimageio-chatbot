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
from simpleaichat import AIChat
from langchain.document_loaders import TextLoader

# # loader = TextLoader("/home/alalulu/workspace/hypha-bot/scripts/imageJ_doc.txt")
# loader = TextLoader("/home/alalulu/workspace/hypha-bot/scripts/imageJ_doc.txt")  

# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)

# embeddings = OpenAIEmbeddings()
# docs_store = Chroma.from_documents(
#     texts, embeddings, collection_name="state-of-union"
# )


# vectorstore_info = VectorStoreInfo(
#     name="imageJ_database",
#     description="documentation of imageJ",
#     vectorstore=docs_store,
# )
# toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
# agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True)

# # response = agent_executor.run(
# #     "what is imagej"
# # )

# # print(response)
embeddings = OpenAIEmbeddings()
output_dir="docs/vectordb"
docs_store = Chroma(collection_name="bioimage.io-docs", persist_directory=output_dir, embedding_function=embeddings)


PREFIX = """Your name is BMZ. You are an chatbot designed to answer questions about sets of documents. 
If you are given a set of raw documents retrieved from a search engine in the `Context`, you will answer the question based on the `Context`.
If the question does not seem to be relavant to the `Context`, just return "I don't know" as the answer.
"""


async def start_server(server_url):
    token = await login({"server_url": server_url})
    server = await connect_to_server({"server_url": server_url, "token": token})
    ai = AIChat(system=PREFIX)

    # llm = OpenAI(temperature=0.9)
    
    async def chat(text, api, context=None):
        # response = agent_executor.run(
        #     text
        # )
        docs = docs_store.similarity_search(text)
        raw_docs = []
        for doc in docs:
            # combine all the docs into one string
            raw_docs.append("```markdown\n" + doc.page_content + "\n```")
        raw_docs = "\n".join(raw_docs)
        prompt = f"##Context\n{raw_docs}\n##Question\n{text}\nNow, please anwser the user's question based the context. The response will render as markdown."
        response = ai(prompt)
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

if __name__ == "__main__":
    server_url = "https://ai.imjoy.io"
    loop = asyncio.get_event_loop()
    loop.create_task(start_server(server_url))
    loop.run_forever()