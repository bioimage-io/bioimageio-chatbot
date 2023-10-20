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
import requests
import sys
import io
import json


def load_model_info():
    response = requests.get("https://bioimage-io.github.io/collection-bioimage-io/collection.json")
    assert response.status_code == 200
    model_info = response.json()
    resource_items = model_info['collection']
    return resource_items


def execute_code(script, context=None):
    if context is None:
        context = {}

    # Redirect stdout and stderr to capture their output
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    try:
        # Create a copy of the context to avoid modifying the original
        local_vars = context.copy()

        # Execute the provided Python script with access to context variables
        exec(script, local_vars)

        # Capture the output from stdout and stderr
        stdout_output = sys.stdout.getvalue()
        stderr_output = sys.stderr.getvalue()

        return {
            "stdout": stdout_output,
            "stderr": stderr_output,
            "context": local_vars  # Include context variables in the result
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "context": context  # Include context variables in the result even if an error occurs
        }
    finally:
        # Restore the original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr


class ModelZooInfoScriptResults(BaseModel):
    """Results of executing the model zoo info query script."""
    stdout: str = Field(description="The output from stdout.")
    stderr: str = Field(description="The output from stderr.")
    request: str = Field(description="User's request in details")
    user_info: str = Field(description="User info for personalize response.")

class DocumentRetrievalInput(BaseModel):
    """Input for finding relevant documents."""
    query: str = Field(description="Query used to retrieve related documents.")
    request: str = Field(description="User's request in details")
    user_info: str = Field(description="Brief user info summary for personalized response, including name, background etc.")

class DirectResponse(BaseModel):
    """Direct response to a user's question."""
    response: str = Field(description="The response to the user's question.")

class DocumentSearchInput(BaseModel):
    """Results of document retrieval from documentation."""
    user_question: str = Field(description="The user's original question.")
    relevant_context: List[str] = Field(description="Context chunks from the documentation (in markdown format), ordered by relevance.")
    user_info: str = Field(description="User info for personalize response.")


class FinalResponse(BaseModel):
    """The final response to the user's question."""
    response: str = Field(description="The answer to the user's question in markdown format. If the question isn't relevant, return 'I don't know'.")


class UserProfile(BaseModel):
    """The user's profile. This will be used to personalize the response."""
    name: str = Field(description="The user's name.", max_length=32)
    occupation: str = Field(description="The user's occupation. ", max_length=128)
    background: str = Field(description="The user's background. ", max_length=256)

class QuestionWithHistory(BaseModel):
    """The user's question, chat history and user's profile."""
    question: str = Field(description="The user's question.")
    chat_history: Optional[List[Dict[str, str]]] = Field(None, description="The chat history.")
    user_profile: Optional[UserProfile] = Field(None, description="The user's profile. You should use this to personalize the response based on the user's background and occupation.")

def create_customer_service(channel):
    resource_items = load_model_info()
    types = set()
    tags = set()
    for resource in resource_items:
        types.add(resource['type'])
        tags.update(resource['tags'])
    types = list(types)
    tags = list(tags)[:10]
    
    resource_item_stats = f"""Each item contains the following fields: {list(resource_items[0].keys())}\nThe available resource types are {types}\nSome example tags: {tags}\nHere is an example: {resource_items[0]}"""
    class ModelZooInfoScript(BaseModel):
        """Create a Python Script to get information about details of models."""
        script: str = Field(description="The script to be executed, the script use a predefined local variable `resources` which contains a list of dictionaries with all the resources in the model zoo (with different types including models and applications etc.), the response to the query should be printed to the stdout. Details about the `resources`:\n" + resource_item_stats)
        request: str = Field(description="User's request in details")
        user_info: str = Field(description="Brief user info summary for personalized response, including name, background etc.")


    docs_store = load_bioimageio_docs(channel)
    assert docs_store._collection.count() > 0

    async def respond_to_user(question_with_history: QuestionWithHistory = None, role: Role = None) -> str:
        """Answer the user's question directly or retrieve relevant documents from the documentation, or create a Python Script to get information about details of models."""
        inputs = [question_with_history.user_profile] + list(question_with_history.chat_history) + [question_with_history.question] 
        request = await role.aask(inputs, Union[DirectResponse, DocumentRetrievalInput, ModelZooInfoScript])
        if isinstance(request, DirectResponse):
            return request.response
        elif isinstance(request, DocumentRetrievalInput):
            relevant_docs = docs_store.similarity_search(request.query, k=2)
            raw_docs = [doc.page_content for doc in relevant_docs]
            search_input = DocumentSearchInput(user_question=request.request, relevant_context=raw_docs, user_info=request.user_info)
            response = await role.aask(search_input, FinalResponse)
            return response.response
        elif isinstance(request, ModelZooInfoScript):
            result = execute_code(request.script, {"resources": resource_items})

            response = await role.aask(ModelZooInfoScriptResults(
                stdout=result["stdout"],
                stderr=result["stderr"],
                request=request.request,
                user_info=request.user_info
            ), FinalResponse)
            return response.response
        
    CustomerServiceRole = Role.create(
        name="Liza",
        profile="Customer Service",
        goal="You are a customer service representative for the help desk of BioImage Model Zoo website. You will answer user's questions about the website, ask for clarification, and retrieve documents from the website's documentation. You may also get user's profile to personalize the response in order to improve the user experience.",
        constraints=None,
        actions=[respond_to_user],
    )
    customer_service = CustomerServiceRole()
    return customer_service

def load_bioimageio_docs(collection_name="bioimage.io-main"):
    # Load from vector store
    embeddings = OpenAIEmbeddings()
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../docs/", collection_name)
    docs_store = Chroma(collection_name=collection_name, persist_directory=output_dir, embedding_function=embeddings)
    return docs_store

async def main():
    customer_service = create_customer_service()
    chat_history=[]
    question = "How can I test the models?"
    profile = UserProfile(name="lulu", occupation="data scientist", background="machine learning and AI")
    m = QuestionWithHistory(question=question, chat_history=chat_history, user_profile=UserProfile.parse_obj(profile))
    resp = await customer_service.handle(Message(content=m.json(), instruct_content=m , role="User"))

    question2 = "What are Model Contribution Guidelines?"
    m = QuestionWithHistory(question=question2, chat_history=chat_history, user_profile=UserProfile.parse_obj(profile))
    resp = await customer_service.handle(Message(content=m.json(), instruct_content=m , role="User"))
    print(resp)
    # resp = await customer_service.handle(Message(content="What are Model Contribution Guidelines?", role="User"))


async def start_server(server_url):
    token = await login({"server_url": server_url})
    server = await connect_to_server({"server_url": server_url, "token": token, "method_timeout": 100})
    # llm = OpenAI(temperature=0.9)
    
    async def chat(text, chat_history, user_profile=None, channel=None, context=None):
        ai = create_customer_service(channel)
        # user_profile = {"name": "lulu", "occupation": "data scientist", "background": "machine learning and AI"}
        m = QuestionWithHistory(question=text, chat_history=chat_history, user_profile=UserProfile.parse_obj(user_profile))
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
        "chat": chat,
        "channels": ["bioimage.io-main", "ImJoy-master"]
    })
    print("visit this to test the bot: https://jsfiddle.net/gzyradL5/11/show")

if __name__ == "__main__":
    # asyncio.run(main())
    server_url = "https://ai.imjoy.io"
    loop = asyncio.get_event_loop()
    loop.create_task(start_server(server_url))
    loop.run_forever()