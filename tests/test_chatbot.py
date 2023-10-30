import os
from bioimageio_chatbot.chatbot import create_customer_service, QuestionWithHistory, UserProfile
from schema_agents.schema import Message
import pytest

KNOWLEDGE_BASE_PATH = "./bioimageio-knowledge-base"

@pytest.mark.asyncio
async def test_chatbot():

    customer_service = create_customer_service(KNOWLEDGE_BASE_PATH)
    chat_history=[]

    question = "Which tool can I use to analyse western blot image?"
    profile = UserProfile(name="lulu", occupation="data scientist", background="machine learning and AI")
    m = QuestionWithHistory(question=question, chat_history=chat_history, user_profile=UserProfile.parse_obj(profile), channel_id=None)
    resp = await customer_service.handle(Message(content=m.json(), data=m , role="User"))
    assert resp

    question = "Which tool can I use to segment an cell image?"
    profile = UserProfile(name="lulu", occupation="data scientist", background="machine learning and AI")
    m = QuestionWithHistory(question=question, chat_history=chat_history, user_profile=UserProfile.parse_obj(profile), channel_id=None)
    resp = await customer_service.handle(Message(content=m.json(), data=m , role="User"))
    assert resp
    
    question = "How can I test the models?"
    profile = UserProfile(name="lulu", occupation="data scientist", background="machine learning and AI")
    m = QuestionWithHistory(question=question, chat_history=chat_history, user_profile=UserProfile.parse_obj(profile), channel_id="bioimage.io")
    resp = await customer_service.handle(Message(content=m.json(), data=m , role="User"))
    assert resp

    question = "What are Model Contribution Guidelines?"
    m = QuestionWithHistory(question=question, chat_history=chat_history, user_profile=UserProfile.parse_obj(profile))
    resp = await customer_service.handle(Message(content=m.json(), data=m , role="User"))
    assert resp
