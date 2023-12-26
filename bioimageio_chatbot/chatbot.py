import asyncio
import os
import json
import datetime
import secrets
import aiofiles
from imjoy_rpc.hypha import login, connect_to_server
from pathlib import Path

from pydantic import BaseModel, Field
from schema_agents.role import Role
from schema_agents.schema import Message
from typing import Any, Dict, List, Optional, Union
import sys
import io
from bioimageio_chatbot.knowledge_base import load_knowledge_base
from bioimageio_chatbot.utils import get_manifest
from bioimageio_chatbot.biii import search_biii_with_links, BiiiRow
import pkg_resources
from bioimageio_chatbot.jsonschema_pydantic import jsonschema_to_pydantic, JsonSchemaObject


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
            # "context": local_vars  # Include context variables in the result
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            # "context": context  # Include context variables in the result even if an error occurs
        }
    finally:
        # Restore the original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

class BiiiQuery(BaseModel):
    """Generate Query to search on biii.eu."""
    keywords: List[str] = Field(description="A list of search keywords, no space allowed in each keyword.")
    top_k: int = Field(10, description="The maximum number of search results to return. Should use a small number to avoid overwhelming the user.")

class BiiiSearchResult(BaseModel):
    """Search results from biii.eu"""
    results: List[BiiiRow] = Field(description="Search results from biii.eu")
    base_url: str = Field(description="The based URL of the search results, e.g. ImageJ (/imagej) will become <base_url>/imagej")

class DirectResponse(BaseModel):
    """Direct response to a user's question if you are confident about the answer."""
    response: str = Field(description="The response to the user's question answering what that asked for.")

class LearningResponse(BaseModel):
    """Pedagogical response to user's question if the user's question is related to learning, the response should include such as key terms and important concepts about the topic. Only use this response when user is asking for general information about a topic."""
    response: str = Field(description="The response to user's question, make sure the response is pedagogical and educational.")

class CodingResponse(BaseModel):
    """If user's question is related to scripting or coding in bioimage analysis, generate code to help user to create valid scripts or understand code. Only use this response when user is asking for coding help."""
    response: str = Field(description="The response to user's question, make sure the response contains valid code, with concise explaination.")

class DocWithScore(BaseModel):
    """A document with an associated relevance score."""
    doc: str = Field(description="The document retrieved.")
    score: float = Field(description="The relevance score of the retrieved document.")
    metadata: Dict[str, Any] = Field(description="The document's metadata.")
    base_url: Optional[str] = Field(None, description="The documentation's base URL, which will be used to resolve the relative URLs in the retrieved document chunks when producing markdown links.")

class DocumentSearchInput(BaseModel):
    """Results of a document retrieval process from a documentation base."""
    relevant_context: List[DocWithScore] = Field(description="Chunks of context retrieved from the documentation that are relevant to the user's original question.")
    format: Optional[str] = Field(None, description="The format of the document.")

class UserProfile(BaseModel):
    """The user's profile. This will be used to personalize the response to the user."""
    name: str = Field(description="The user's name.", max_length=32)
    occupation: str = Field(description="The user's occupation.", max_length=128)
    background: str = Field(description="The user's background.", max_length=256)

class ExtensionCallInput(BaseModel):
    """Result of calling an extension function"""
    result: Optional[Any] = Field(None, description="The result of calling the extension function.")
    error: Optional[str] = Field(None, description="The error message if the extension function call failed.")

class DocumentResponse(BaseModel):
    """The Document Response to the user's question based on the documentation search results.
    Make sure the response is relevant to the user's question, do not provide irrelevant information to the user. The response should be tailored to uer's info if provided. 
    If the documentation search results are relevant to the user's question, provide a text response to the question based on the search results.
    If the documentation search results contains only low relevance scores, return 'I don't know'.
    Importantly, if you can't confidently provide a relevant response to the user's question, return 'Sorry I didn't find relevant information, please try again.'."""
    response: str = Field(description="The answer to the user's question based on the search results. Can be either a detailed response in markdown format if the search results are relevant to the user's question or 'I don't know'.")


class BiiiResponse(BaseModel):
    """Summarize the search results from biii.eu. Make sure the response is relevant to the user's question, do not provide irrelevant information to the user."""
    response: str = Field(description="The answer to the user's question based on the search results. Can be either a detailed response in markdown format if the search results are relevant to the user's question or 'I don't know'. It should resolve relative URLs in the search results using the base_url.")

class ExtensionCallResponse(BaseModel):
    """Summarize the result of calling an extension function"""
    response: str = Field(description="The answer to the user's question based on the result of calling the extension function.")

class ChannelInfo(BaseModel):
    """The selected knowledge base channel for the user's question. If provided, rely only on the selected channel when answering the user's question."""
    id: str = Field(description="The channel id.")
    name: str = Field(description="The channel name.")
    description: str = Field(description="The channel description.")

class QuestionWithHistory(BaseModel):
    """The user's question, chat history, and user's profile."""
    question: str = Field(description="The user's question.")
    chat_history: Optional[List[Dict[str, str]]] = Field(None, description="The chat history.")
    user_profile: Optional[UserProfile] = Field(None, description="The user's profile. You should use this to personalize the response based on the user's background and occupation.")
    channel_info: Optional[ChannelInfo] = Field(None, description="The selected channel of the user's question. If provided, rely only on the selected channel when answering the user's question.")
    image_data: Optional[str] = Field(None, description = "The uploaded image data if the user has uploaded an image. If provided, run a cellpose task")
    chatbot_extensions: Optional[List[Dict[str, Any]]] = Field(None, description="Chatbot extensions.")

class ResponseStep(BaseModel):
    """Response step"""
    name: str = Field(description="Step name")
    details: Optional[dict] = Field(None, description="Step details")

class RichResponse(BaseModel):
    """Rich response with text and intermediate steps"""
    text: str = Field(description="Response text")
    steps: List[ResponseStep] = Field(description="Intermediate steps")

class ResponseStep(BaseModel):
    """Response step"""
    name: str = Field(description="Step name")
    details: Optional[dict] = Field(None, description="Step details")

class RichResponse(BaseModel):
    """Rich response with text and intermediate steps"""
    text: str = Field(description="Response text")
    steps: List[ResponseStep] = Field(description="Intermediate steps")


def create_customer_service(db_path):
    collections = get_manifest()['collections']
    docs_store_dict = load_knowledge_base(db_path)
    collection_info_dict = {collection['id']: collection for collection in collections}
    
    channels_info = "\n".join(f"""- `{collection['id']}`: {collection['description']}""" for collection in collections)

    async def respond_to_user(question_with_history: QuestionWithHistory = None, role: Role = None) -> str:
        """Answer the user's question directly or retrieve relevant documents from the documentation, or create a Python Script to get information about details of models."""
        steps = []
        inputs = [question_with_history.user_profile] + list(question_with_history.chat_history) + [question_with_history.question]
        # The channel info will be inserted at the beginning of the inputs
        if question_with_history.channel_info:
            inputs.insert(0, question_with_history.channel_info)
        if question_with_history.channel_info:
            channel_prompt = f"The channel_id of the knowledge base to search. It MUST be set to {question_with_history.channel_info.id} (selected by the user)."
        else:
            channel_prompt = f"The channel_id of the knowledge base to search. It MUST be the same as the user provided channel_id. If not specified, either select 'all' to search through all the available knowledge base channels or select one from the available channels which are:\n{channels_info}. If you are not sure which channel to select, select 'all'."
        
        
        class DocumentRetrievalInput(BaseModel):
            """Input for searching knowledge bases and finding documents relevant to the user's request."""
            query: str = Field(description="The query used to retrieve documents related to the user's request.")
            channel_id: str = Field(description=channel_prompt)

        async def retrieve(req: DocumentRetrievalInput) -> DocumentSearchInput:
            """Retrieve documents from the documentation based on the user's question."""
            if question_with_history.channel_info:
                req.channel_id = question_with_history.channel_info.id

            steps.append(ResponseStep(name="Document Retrieval", details=req.dict()))
            if req.channel_id == "all":
                docs_with_score = []
                # loop through all the channels
                for channel_id in docs_store_dict.keys():
                    docs_store = docs_store_dict[channel_id]
                    collection_info = collection_info_dict[channel_id]
                    base_url = collection_info.get('base_url')
                    print(f"Retrieving documents from database {channel_id} with query: {req.query}")
                    results_with_scores = await docs_store.asimilarity_search_with_relevance_scores(req.query, k=2)
                    new_docs_with_score = [DocWithScore(doc=doc.page_content, score=score, metadata=doc.metadata, base_url=base_url) for doc, score in results_with_scores]
                    docs_with_score.extend(new_docs_with_score)
                    print(f"Retrieved documents:\n{new_docs_with_score[0].doc[:20] + '...'} (score: {new_docs_with_score[0].score})\n{new_docs_with_score[1].doc[:20] + '...'} (score: {new_docs_with_score[1].score})")
                # rank the documents by relevance score
                docs_with_score = sorted(docs_with_score, key=lambda x: x.score, reverse=True)
                # only keep the top 3 documents
                docs_with_score = docs_with_score[:3]
                
                search_input = DocumentSearchInput(relevant_context=docs_with_score, format=None)
                steps.append(ResponseStep(name="Document Search", details=search_input.dict()))
                
            else:
                docs_store = docs_store_dict[req.channel_id]
                collection_info = collection_info_dict[req.channel_id]
                base_url = collection_info.get('base_url')
                print(f"Retrieving documents from database {req.channel_id} with query: {req.query}")
                results_with_scores = await docs_store.asimilarity_search_with_relevance_scores(req.query, k=3)
                docs_with_score = [DocWithScore(doc=doc.page_content, score=score, metadata=doc.metadata, base_url=base_url) for doc, score in results_with_scores]
                print(f"Retrieved documents:\n{docs_with_score[0].doc[:20] + '...'} (score: {docs_with_score[0].score})\n{docs_with_score[1].doc[:20] + '...'} (score: {docs_with_score[1].score})\n{docs_with_score[2].doc[:20] + '...'} (score: {docs_with_score[2].score})")
                search_input = DocumentSearchInput(relevant_context=docs_with_score, format=collection_info.get('format'))
                steps.append(ResponseStep(name="Document Search", details=search_input.dict()))
            return search_input
            
        async def biii_search(req: BiiiQuery) -> Union[BiiiSearchResult, None]:
            """Search software tools on BioImage Informatics Index (biii.eu) is a platform for sharing bioimage analysis software and tools."""
            print(f"Searching biii.eu with keywords: {req.keywords}, top_k: {req.top_k}")
            try:
                loop = asyncio.get_running_loop()
                steps.append(ResponseStep(name="Search on biii.eu", details=req.dict()))
                results = await loop.run_in_executor(None, search_biii_with_links, req.keywords, "software", "")
                if results:
                    results = BiiiSearchResult(results=results[:req.top_k], user_info=req.user_info, base_url="https://biii.eu")
                    steps.append(ResponseStep(name="Summarize results from biii.eu", details=results.dict()))
                    return results
                else:
                    return None
            except Exception as e:
                return None

        available_tools = channels_info + "\n- `biii.eu`: " + biii_search.__doc__
        
        class UserResponse(BaseModel):
            """Response to a user's question."""
            use_tools_and_retrieval: bool = Field(description="Whether to use document retrieval or tools to answer the user's question. If you are confident about the answer without additional information, set this to False.")
            response: Optional[str] = Field(description="Response to the user's question. Set to None if you want to use document retrieval or tools to answer the user's question.")
            
        UserResponse.__doc__ += f"If the question is related to the following, use retrieval and tools to improve your anwser:\n{available_tools}"
        
        resp = await role.aask(inputs, UserResponse)
        steps.append(ResponseStep(name="User Response", details=resp.dict()))
        if resp.use_tools_and_retrieval:
            # inputs.append("You can select one or multiple response functions to answer the user's question.")
            # inputs.append("Preliminary response: " + resp.response)
            class CallResult(BaseModel):
                """Result of calling a response function"""
                response: Optional[str] = Field(description="The response to the user's question. If the call results are relevant to the user's question, the response should summarize them. Otherwise, set response to None")
                continue_search: bool = Field(description="Set to True if you want to continue searching for more relevant information.")

            async def find_anwser():
                call_resp = await role.acall(inputs, [retrieve, biii_search], CallResult) # Union[DocumentResponse, BiiiResponse])
                steps.append(ResponseStep(name="Call Result", details=call_resp.dict()))
                if call_resp.continue_search:
                    inputs.append(call_resp)
                    return find_anwser()
                else:
                    return call_resp.response
                
            return RichResponse(text=await find_anwser(), steps=steps)
        else:
            return RichResponse(text=resp.response, steps=steps)
        
        # builtin_response_types = [DirectResponse, DocumentRetrievalInput, SearchOnBiii, LearningResponse, CodingResponse]
        # extension_types = [mode_d['schema_class'] for mode_d in question_with_history.chatbot_extensions] if question_with_history.chatbot_extensions else []
        # response_types = tuple(builtin_response_types + extension_types)
        
        # class Planning(BaseModel):
        #     """Plan which responses to use based on the user's question. Select one or more responses from the available response types."""
        #     reasoning: str = Field(description="The reasoning behind the selected response types.")
        #     select_response_types: List[str] = Field(description="The response types to use. The available response types are: " + ", ".join([type.__name__ for type in response_types]))
        # plan_response = await role.aask(inputs, Planning)
        
        # resp_list = []
        # for response_type in plan_response.select_response_types:
            
        #     if response_type == "DirectResponse":
        #         resp = await role.aask(inputs, DirectResponse)
        #         steps.append(ResponseStep(name=type(resp).__name__, details=resp.dict()))
        #         # return RichResponse(text=resp.response, steps=steps)
        #     elif response_type == "SearchOnBiii":
        #         req = await role.aask(inputs, SearchOnBiii)
        #         print(f"Searching biii.eu with keywords: {req.keywords}, top_k: {req.top_k}")
        #         try:
        #             loop = asyncio.get_running_loop()
        #             steps.append(ResponseStep(name="Search on biii.eu", details=req.dict()))
        #             results = await loop.run_in_executor(None, search_biii_with_links, req.keywords, "software", "")
        #             if results:
        #                 results = BiiiSearchResult(results=results[:req.top_k], request=req.request, user_info=req.user_info, base_url="https://biii.eu", preliminary_response=req.preliminary_response)
        #                 steps.append(ResponseStep(name="Summarize results from biii.eu", details=results.dict()))
        #                 resp = await role.aask(results, BiiiResponse)
        #                 # return RichResponse(text=resp.response, steps=steps)
        #             # else:
        #             #     return RichResponse(text=f"Sorry I didn't find relevant information in biii.eu about {req.keywords}", steps=steps)
        #         except Exception as e:
        #             return RichResponse(text=f"Failed to search biii.eu, error: {e}", steps=steps)
        #     elif response_type == "DocumentRetrievalInput":
        #         req = await role.aask(inputs, DocumentRetrievalInput)
        #         if question_with_history.channel_info:
        #             req.channel_id = question_with_history.channel_info.id

        #         steps.append(ResponseStep(name="Document Retrieval", details=req.dict()))
        #         if req.channel_id == "all":
        #             docs_with_score = []
        #             # loop through all the channels
        #             for channel_id in docs_store_dict.keys():
        #                 docs_store = docs_store_dict[channel_id]
        #                 collection_info = collection_info_dict[channel_id]
        #                 base_url = collection_info.get('base_url')
        #                 print(f"Retrieving documents from database {channel_id} with query: {req.query}")
        #                 results_with_scores = await docs_store.asimilarity_search_with_relevance_scores(req.query, k=2)
        #                 new_docs_with_score = [DocWithScore(doc=doc.page_content, score=score, metadata=doc.metadata, base_url=base_url) for doc, score in results_with_scores]
        #                 docs_with_score.extend(new_docs_with_score)
        #                 print(f"Retrieved documents:\n{new_docs_with_score[0].doc[:20] + '...'} (score: {new_docs_with_score[0].score})\n{new_docs_with_score[1].doc[:20] + '...'} (score: {new_docs_with_score[1].score})")
        #             # rank the documents by relevance score
        #             docs_with_score = sorted(docs_with_score, key=lambda x: x.score, reverse=True)
        #             # only keep the top 3 documents
        #             docs_with_score = docs_with_score[:3]
                    
        #             search_input = DocumentSearchInput(user_question=req.request, relevant_context=docs_with_score, user_info=req.user_info, format=None, preliminary_response=req.preliminary_response)
        #             steps.append(ResponseStep(name="Document Search", details=search_input.dict()))
        #             resp = await role.aask(search_input, DocumentResponse)
        #             steps.append(ResponseStep(name="Document Response", details=resp.dict()))
        #         else:
        #             docs_store = docs_store_dict[req.channel_id]
        #             collection_info = collection_info_dict[req.channel_id]
        #             base_url = collection_info.get('base_url')
        #             print(f"Retrieving documents from database {req.channel_id} with query: {req.query}")
        #             results_with_scores = await docs_store.asimilarity_search_with_relevance_scores(req.query, k=3)
        #             docs_with_score = [DocWithScore(doc=doc.page_content, score=score, metadata=doc.metadata, base_url=base_url) for doc, score in results_with_scores]
        #             print(f"Retrieved documents:\n{docs_with_score[0].doc[:20] + '...'} (score: {docs_with_score[0].score})\n{docs_with_score[1].doc[:20] + '...'} (score: {docs_with_score[1].score})\n{docs_with_score[2].doc[:20] + '...'} (score: {docs_with_score[2].score})")
        #             search_input = DocumentSearchInput(user_question=req.request, relevant_context=docs_with_score, user_info=req.user_info, format=collection_info.get('format'), preliminary_response=req.preliminary_response)
        #             steps.append(ResponseStep(name="Document Search", details=search_input.dict()))
        #             resp = await role.aask(search_input, DocumentResponse)
        #             steps.append(ResponseStep(name="Document Response", details=resp.dict()))
        #         # source: doc.metadata.get('source', 'N/A')
        #         # return RichResponse(text=resp.response, steps=steps)
        #     elif isinstance(req, tuple(extension_types)):
        #         idx = extension_types.index(type(req))
        #         mode_d = question_with_history.chatbot_extensions[idx]
        #         response_function = mode_d['execute']
        #         mode_name = mode_d['name']
        #         steps.append(ResponseStep(name="Extension: " + mode_name, details=req.dict()))
        #         try:
        #             result = await response_function(req.dict())
        #             steps.append(ResponseStep(name="Summarize result: " + mode_name, details={"result": result}))
        #             resp = await role.aask(ExtensionCallInput(user_question=question_with_history.question, user_profile=question_with_history.user_profile, result=result), ExtensionCallResponse)
        #             steps.append(ResponseStep(name="Result: " + mode_name, details=resp.dict()))
        #             return RichResponse(text=resp.response, steps=steps)
        #         except Exception as e:
        #             resp = await role.aask(ExtensionCallInput(user_question=question_with_history.question, user_profile=question_with_history.user_profile, error=str(e)), ExtensionCallResponse)
        #             steps.append(ResponseStep(name="Result: " + mode_name, details=resp.dict()))
        #             # return RichResponse(text=resp.response, steps=steps)
        #     else:
        #         raise ValueError(f"Unknown response type: {type(req)}")

        #     resp_list.append(resp)
        
        # class SummaryResponse(BaseModel):
        #     """Summarize responses from different response types, make sure the response is relevant to the user's question. If the response is not relevant, do not include it in the combined response."""
        #     response: str = Field(description="The summarized response from single or multiple responses.")
        # final_resp = await role.aask(resp_list, SummaryResponse)
        # return RichResponse(text=final_resp.response, steps=steps)
        
    customer_service = Role(
        name="Melman",
        profile="Customer Service",
        goal="Your goal as Melman from Madagascar, the community knowledge base manager, is to assist users in effectively utilizing the knowledge base for bioimage analysis. You are responsible for answering user questions, providing clarifications, retrieving relevant documents, and executing scripts as needed. Your overarching objective is to make the user experience both educational and enjoyable.",
        constraints=None,
        actions=[respond_to_user],
    )
    return customer_service

async def save_chat_history(chat_log_full_path, chat_his_dict):    
    # Serialize the chat history to a json string
    chat_history_json = json.dumps(chat_his_dict)

    # Write the serialized chat history to the json file
    async with aiofiles.open(chat_log_full_path, mode='w', encoding='utf-8') as f:
        await f.write(chat_history_json)

    
async def connect_server(server_url):
    """Connect to the server and register the chat service."""
    login_required = os.environ.get("BIOIMAGEIO_LOGIN_REQUIRED") == "true"
    if login_required:
        token = await login({"server_url": server_url})
    else:
        token = None
    server = await connect_to_server({"server_url": server_url, "token": token, "method_timeout": 100})
    await register_chat_service(server)
    
async def register_chat_service(server):
    """Hypha startup function."""
    manifest = get_manifest()
    collections = manifest['collections']
    additional_channels = manifest['additional_channels']
    login_required = os.environ.get("BIOIMAGEIO_LOGIN_REQUIRED") == "true"
    knowledge_base_path = os.environ.get("BIOIMAGEIO_KNOWLEDGE_BASE_PATH", "./bioimageio-knowledge-base")
    assert knowledge_base_path is not None, "Please set the BIOIMAGEIO_KNOWLEDGE_BASE_PATH environment variable to the path of the knowledge base."
    if not os.path.exists(knowledge_base_path):
        print(f"The knowledge base is not found at {knowledge_base_path}, will download it automatically.")
        os.makedirs(knowledge_base_path, exist_ok=True)

    chat_logs_path = os.environ.get("BIOIMAGEIO_CHAT_LOGS_PATH", "./chat_logs")
    assert chat_logs_path is not None, "Please set the BIOIMAGEIO_CHAT_LOGS_PATH environment variable to the path of the chat logs folder."
    if not os.path.exists(chat_logs_path):
        print(f"The chat session folder is not found at {chat_logs_path}, will create one now.")
        os.makedirs(chat_logs_path, exist_ok=True)
    
    channel_id_by_name = {collection['name']: collection['id'] for collection in collections + additional_channels}
    description_by_id = {collection['id']: collection['description'] for collection in collections + additional_channels}
    customer_service = create_customer_service(knowledge_base_path)
    event_bus = customer_service.get_event_bus()
    event_bus.register_default_events()
        
    def load_authorized_emails():
        if login_required:
            authorized_users_path = os.environ.get("BIOIMAGEIO_AUTHORIZED_USERS_PATH")
            if authorized_users_path:
                assert os.path.exists(authorized_users_path), f"The authorized users file is not found at {authorized_users_path}"
                with open(authorized_users_path, "r") as f:
                    authorized_users = json.load(f)["users"]
                authorized_emails = [user["email"] for user in authorized_users if "email" in user]
            else:
                authorized_emails = None
        else:
            authorized_emails = None
        return authorized_emails

    authorized_emails = load_authorized_emails()
    def check_permission(user):
        if authorized_emails is None or user["email"] in authorized_emails:
            return True
        else:
            return False
        
    async def report(user_report, context=None):
        if login_required and context and context.get("user"):
            assert check_permission(context.get("user")), "You don't have permission to report the chat history."
        # get the chatbot version
        version = pkg_resources.get_distribution('bioimageio-chatbot').version
        chat_his_dict = {'type':user_report['type'],
                         'feedback':user_report['feedback'],
                         'conversations':user_report['messages'], 
                         'session_id':user_report['session_id'], 
                        'timestamp': str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), 
                        'user': context.get('user'),
                        'version': version}
        session_id = user_report['session_id'] + secrets.token_hex(4)
        filename = f"report-{session_id}.json"
        # Create a chat_log.json file inside the session folder
        chat_log_full_path = os.path.join(chat_logs_path, filename)
        await save_chat_history(chat_log_full_path, chat_his_dict)
        print(f"User report saved to {filename}")
        
    async def chat(text, chat_history, user_profile=None, channel=None, status_callback=None, session_id=None, image_data=None, extensions=None, context=None):
        if login_required and context and context.get("user"):
            assert check_permission(context.get("user")), "You don't have permission to use the chatbot, please sign up and wait for approval"
        session_id = session_id or secrets.token_hex(8)
        # Listen to the `stream` event
        async def stream_callback(message):
            if message.type in ["function_call", "text"]:
                if message.session.id == session_id:
                    await status_callback(message.dict())

        event_bus.on("stream", stream_callback)
        
        if extensions is not None: 
            chatbot_extensions = []
            for ext in extensions:
                schema = await ext['get_schema']()
                chatbot_extensions.append(
                    {
                        "name": ext['name'],
                        'description': ext['description'],
                        "schema_class": jsonschema_to_pydantic(JsonSchemaObject.parse_obj(schema)),
                        "execute": ext['execute']
                    }
                )

        if image_data:
            await status_callback({"type": "text", "content": f"\n![Uploaded Image]({image_data})\n"})

        # Get the channel id by its name
        if channel == 'auto':
            channel = None
        if channel:
            assert channel in channel_id_by_name, f"Channel {channel} is not found, available channels are {list(channel_id_by_name.keys())}"
            channel_id = channel_id_by_name[channel]
        else:
            channel_id = None
        
        channel_info = channel_id and {"id": channel_id, "name": channel, "description": description_by_id[channel_id]}
        if channel_info:
            channel_info = ChannelInfo.parse_obj(channel_info)
        
        
        m = QuestionWithHistory(question=text, chat_history=chat_history, user_profile=UserProfile.parse_obj(user_profile),channel_info=channel_info, chatbot_extensions=chatbot_extensions)
        if image_data: # Checks if user uploaded an image, if so wait for it to upload
            await status_callback({"type": "text", "content": "Uploading image..."})
            m.image_data = image_data
            if text == "":
                m.question = "Please analyze this image" # In case user uploaded image with no text message
        try:
            response = await customer_service.handle(Message(content="", data=m , role="User", session_id=session_id))
            # get the content of the last response
            response = response[-1].data # type: RichResponse
            print(f"\nUser: {text}\nChatbot: {response.text}")
        except Exception as e:
            event_bus.off("stream", stream_callback)
            raise e
        else:
            event_bus.off("stream", stream_callback)

        if session_id:
            chat_history.append({ 'role': 'user', 'content': text })
            chat_history.append({ 'role': 'assistant', 'content': response.text })
            version = pkg_resources.get_distribution('bioimageio-chatbot').version
            chat_his_dict = {'conversations':chat_history, 
                     'timestamp': str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), 
                     'user': context.get('user'),
                     'version': version}
            filename = f"chatlogs-{session_id}.json"
            chat_log_full_path = os.path.join(chat_logs_path, filename)
            await save_chat_history(chat_log_full_path, chat_his_dict)
            print(f"Chat history saved to {filename}")
        return response.dict()

    async def ping(context=None):
        if login_required and context and context.get("user"):
            assert check_permission(context.get("user")), "You don't have permission to use the chatbot, please sign up and wait for approval"
        return "pong"

    channels = [collection['name'] for collection in collections + additional_channels]
    hypha_service_info = await server.register_service({
        "name": "BioImage.IO Chatbot",
        "id": "bioimageio-chatbot",
        "config": {
            "visibility": "public",
            "require_context": True
        },
        "ping": ping,
        "chat": chat,
        "report": report,
        "channels": channels,
    })
    
    version = pkg_resources.get_distribution('bioimageio-chatbot').version
    def reload_index():
        with open(os.path.join(os.path.dirname(__file__), "static/index.html"), "r") as f:
            index_html = f.read()
        index_html = index_html.replace("https://ai.imjoy.io", server.config['public_base_url'] or f"http://127.0.0.1:{server.config['port']}")
        index_html = index_html.replace('"bioimageio-chatbot"', f'"{hypha_service_info["id"]}"')
        index_html = index_html.replace('v0.1.0', f'v{version}')
        index_html = index_html.replace("LOGIN_REQUIRED", "true" if login_required else "false")
        return index_html
    
    index_html = reload_index()
    debug = os.environ.get("BIOIMAGEIO_DEBUG") == "true"
    async def index(event, context=None):
        return {
            "status": 200,
            "headers": {'Content-Type': 'text/html'},
            "body": reload_index() if debug else index_html,
        }
    
    await server.register_service({
        "id": "bioimageio-chatbot-client",
        "type": "functions",
        "config": {
            "visibility": "public",
            "require_context": False
        },
        "index": index,
    })
    server_url = server.config['public_base_url']

    user_url = f"{server_url}/{server.config['workspace']}/apps/bioimageio-chatbot-client/index"
    print(f"The BioImage.IO Chatbot is available at: {user_url}")
    
if __name__ == "__main__":
    # asyncio.run(main())
    server_url = """https://ai.imjoy.io"""
    # server_url = "https://hypha.bioimage.io/"
    loop = asyncio.get_event_loop()
    loop.create_task(connect_server(server_url))
    loop.run_forever()