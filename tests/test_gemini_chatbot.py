import os
from bioimageio_chatbot.chatbot import create_assistants, get_builtin_extensions, QuestionWithHistory, UserProfile
from schema_agents.role import Role
from schema_agents.schema import Message
import json
import pytest

KNOWLEDGE_BASE_PATH = "./bioimageio-knowledge-base"

@pytest.fixture
def builtin_extensions():
    return get_builtin_extensions()

@pytest.fixture
def elara(builtin_extensions):
    assistants = create_assistants(builtin_extensions)
    # find an assistant name Elara
    m = [assistant for assistant in assistants if assistant['name'] == "Elara"][0]
    return m['agent']


    
# @pytest.mark.asyncio
# async def test_respond_user_str():
#     async def respond_to_user(query: str, role: Role) -> str:
#         """Respond to user."""
#         response = await role.aask(query, str)
#         return response
    
    
#     role = Role(name="Alice",
#                 profile="Customer service",
#                 goal="Efficiently communicate with the user and translate the user's needs to technical requirements",
#                 constraints=None,
#                 actions=[respond_to_user],
#                 backend="gemini",)
    
#     messages = ["hi"]
#     responses = await role.handle(messages)
#     assert responses
# @pytest.mark.asyncio
# async def test_tool_call(builtin_extensions, elara):
#     # load saved json file 
#     with open('test_messages.json', 'r') as file:
#         test_messages = json.load(file)

#     messages = elara._llm.format_msg(test_messages)
#     assert 

@pytest.mark.asyncio
async def test_chatbot(builtin_extensions, elara):
    select_extensions = [
        {"id": "eurobioimaging"}
    ]
    chat_history=[]
    question = "Which technique can I use to image neurons?"
    profile = UserProfile(name="lulu", occupation="data scientist", background="machine learning and AI")
    m = QuestionWithHistory(question=question, chat_history=chat_history, user_profile=UserProfile.model_validate(profile), channel_id=None, chatbot_extensions=select_extensions)
    resp = await elara.handle(Message(content="", data=m , role="User"))
    assert resp
    str_resp = [str(element) for element in resp]
    assert any(["EurobioimagingSearchTechnology" in element for element in str_resp])
    
    chat_history.append(
                {"role": "user", "content": question}
            )
    chat_history.append(
                {"role": "assistant", "content": resp.text}
            )
    # question2 = "tell me where i can find this technique?"

    
    

