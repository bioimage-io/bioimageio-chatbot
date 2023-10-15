import asyncio
import os
from imjoy_rpc.hypha import login, connect_to_server
from langchain.llms import OpenAI

from pydantic import BaseModel, Field
from schema_agents.role import Role
from schema_agents.schema import Message
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from typing import Any, Dict, List, Optional, Union

class DocumentRetrievalInput(BaseModel):
    """Input for finding relevant documents."""
    query: str = Field(description="Query used to retrieve related documents.")
    request: str = Field(description="User's request in details")

class DirectResponse(BaseModel):
    """Direct response to a user's question."""
    response: str = Field(description="The response to the user's question.")

class DocumentSearchInput(BaseModel):
    """Results of document retrieval from documentation."""
    user_question: str = Field(description="The user's original question.")
    relevant_context: List[str] = Field(description="Context chunks from the documentation (in markdown format), ordered by relevance.")

class FinalResponse(BaseModel):
    """The final response to the user's question."""
    response: str = Field(description="The answer to the user's question in markdown format. If the question isn't relevant, return 'I don't know'.")

class QuestionWithHistory(BaseModel):
    """The user's question and the chat history."""
    question: str = Field(description="The user's question.")
    chat_history: Optional[List[Dict[str, str]]] = Field(None, description="The chat history.")

async def retrieve_document(question_with_history: QuestionWithHistory = None, role: Role = None) -> str:
    """Answer the user's question directly or retrieve relevant documents from the documentation."""
    inputs = list(question_with_history.chat_history) + [question_with_history.question]
    request = await role.aask(inputs, Union[DirectResponse, DocumentRetrievalInput])
    if isinstance(request, DirectResponse):
        return request.response
    relevant_docs = docs_store.similarity_search(request.query, k=2)
    raw_docs = [doc.page_content for doc in relevant_docs]
    search_input = DocumentSearchInput(user_question=request.request, relevant_context=raw_docs)
    response = await role.aask(search_input, FinalResponse)
    return response.response

# Load from vector store
embeddings = OpenAIEmbeddings()
output_dir = "docs/vectordb"
docs_store = Chroma(collection_name="bioimage.io-docs", persist_directory=output_dir, embedding_function=embeddings)

def create_customer_service():
    CustomerServiceRole = Role.create(
        name="Liza",
        profile="Customer Service",
        goal="You are a customer service representative for the help desk of BioImage Model Zoo website. You will answer user's questions about the website, ask for clarification, and retrieve documents from the website's documentation.",
        constraints=None,
        actions=[retrieve_document],
    )
    customer_service = CustomerServiceRole()
    return customer_service

async def main():
    customer_service = create_customer_service()
    resp = await customer_service.handle(Message(content="Who are you?", role="User"))

    resp = await customer_service.handle(Message(content="What are Model Contribution Guidelines?", role="User"))


async def start_server(server_url):
    token = await login({"server_url": server_url})
    server = await connect_to_server({"server_url": server_url, "token": token})
    # llm = OpenAI(temperature=0.9)
    
    async def chat(text, chat_history, context=None):
        ai = create_customer_service()
        m = QuestionWithHistory(question=text, chat_history=chat_history)
        response = await ai.handle(Message(content=m.json(), instruct_content=m , role="User"))
        # get the content of the last response
        response = response[-1].content
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
    print("visit this to test the bot: https://jsfiddle.net/gzyradL5/11/show")

if __name__ == "__main__":
    # asyncio.run(main())
    server_url = "https://ai.imjoy.io"
    loop = asyncio.get_event_loop()
    loop.create_task(start_server(server_url))
    loop.run_forever()