import os
from bioimageio_chatbot.chatbot import create_assistants, get_builtin_extensions, QuestionWithHistory, UserProfile
from schema_agents.schema import Message
import pytest

KNOWLEDGE_BASE_PATH = "./bioimageio-knowledge-base"

@pytest.fixture
def builtin_extensions():
    return get_builtin_extensions()

@pytest.fixture
def melman(builtin_extensions):
    assistants = create_assistants(builtin_extensions)
    # find an assistant name Melman
    m = [assistant for assistant in assistants if assistant['name'] == "Melman"][0]
    return m['agent']

@pytest.mark.asyncio
async def test_chatbot(builtin_extensions, melman):
    select_extensions = [
        {"id": "bioimage_archive"}
    ]
    chat_history=[]
    question = "Which tool can I use to analyse western blot image?"
    profile = UserProfile(name="lulu", occupation="data scientist", background="machine learning and AI")
    m = QuestionWithHistory(question=question, chat_history=chat_history, user_profile=UserProfile.model_validate(profile), channel_id=None, chatbot_extensions=select_extensions)
    resp = await melman.handle(Message(content="", data=m , role="User"))
    assert resp
    str_resp = [str(element) for element in resp]
    assert any(["BioimageArchiveSearch" in element for element in str_resp])

    question = "Which tool can I use to segment an cell image?"
    profile = UserProfile(name="lulu", occupation="data scientist", background="machine learning and AI")
    m = QuestionWithHistory(question=question, chat_history=chat_history, user_profile=UserProfile.model_validate(profile), channel_id=None, chatbot_extensions=select_extensions)
    resp = await melman.handle(Message(content="", data=m , role="User"))
    assert resp
    str_resp = [str(element) for element in resp]
    assert any(["BioimageArchiveSearch" in element for element in str_resp])
    
    question = "How can I test the models?"
    profile = UserProfile(name="lulu", occupation="data scientist", background="machine learning and AI")
    m = QuestionWithHistory(question=question, chat_history=chat_history, user_profile=UserProfile.model_validate(profile), channel_id=None, chatbot_extensions=select_extensions)
    resp = await melman.handle(Message(content="", data=m , role="User"))
    assert resp
    str_resp = [str(element) for element in resp]
    assert any(["BioimageArchiveSearch" in element for element in str_resp])

    question = "What are Model Contribution Guidelines?"
    m = QuestionWithHistory(question=question, chat_history=chat_history, user_profile=UserProfile.model_validate(profile), channel_id=None, chatbot_extensions=select_extensions)
    resp = await melman.handle(Message(content="", data=m , role="User"))
    assert resp
    str_resp = [str(element) for element in resp]
    assert any(["BioimageArchiveSearch" in element for element in str_resp])

    
    # test biii extension
    select_extensions = [
        {"id": "biii"}
    ]
    question = "What bioimage analysis tools are available for quantifying cell migration?"
    profile = UserProfile(name="lulu", occupation="data scientist", background="machine learning and AI")
    m = QuestionWithHistory(question=question, chat_history=chat_history, user_profile=UserProfile.model_validate(profile), channel_id=None, chatbot_extensions=select_extensions)
    resp = await melman.handle(Message(content="", data=m , role="User"))
    assert resp
    str_resp = [str(element) for element in resp]
    assert any(["BiiiSearch" in element for element in str_resp])
    
    question = "Are there any workflows on biii.eu for 3D reconstruction of neuronal networks from electron microscopy images?"
    profile = UserProfile(name="lulu", occupation="data scientist", background="machine learning and AI")
    m = QuestionWithHistory(question=question, chat_history=chat_history, user_profile=UserProfile.model_validate(profile), channel_id=None, chatbot_extensions=select_extensions)
    resp = await melman.handle(Message(content="", data=m , role="User"))
    assert resp
    str_resp = [str(element) for element in resp]
    assert any(["BiiiSearch" in element for element in str_resp])
    
    
    # test image_sc extension
    select_extensions = [
        {"id": "image_sc_forum"}
    ]
    question = "I got a problem, StarDist stops working! help me find it in image.sc forum."
    profile = UserProfile(name="lulu", occupation="data scientist", background="machine learning and AI")
    m = QuestionWithHistory(question=question, chat_history=chat_history, user_profile=UserProfile.model_validate(profile), channel_id=None, chatbot_extensions=select_extensions)
    resp = await melman.handle(Message(content="", data=m , role="User"))
    assert resp
    # make resp a string
    resp = [str(element) for element in resp]
    assert any(["ImageScForumSearch" in element for element in resp])
    assert any(['''posts":''' in element for element in resp])


    # test web extension
    select_extensions = [
        {"id": "web"}
    ]
    question = "I want to know more about the BioImage Archive"
    profile = UserProfile(name="lulu", occupation="data scientist", background="machine learning and AI")
    m = QuestionWithHistory(question=question, chat_history=chat_history, user_profile=UserProfile.model_validate(profile), channel_id=None, chatbot_extensions=select_extensions)
    resp = await melman.handle(Message(content="", data=m , role="User"))
    assert resp
    str_resp = [str(element) for element in resp]
    assert any(["WebSearch" in element for element in str_resp])
    assert any(['''"content": ''' in element for element in str_resp])
    