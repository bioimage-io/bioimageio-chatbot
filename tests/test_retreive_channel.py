import os
from bioimageio_chatbot.chatbot import create_customer_service, QuestionWithHistory, UserProfile
from schema_agents.schema import Message
import pandas as pd
import asyncio
import pytest

KNOWLEDGE_BASE_PATH = "./bioimageio-knowledge-base"
customer_service = create_customer_service(KNOWLEDGE_BASE_PATH)


@pytest.fixture
def eval_questions():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    eval_file = os.path.join(dir_path, "Minimal-Eval-Test-20240111.csv")
    if os.path.exists(eval_file):
        query_answer = pd.read_csv(eval_file)
    else:
        query_answer = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vTVgE2_eqBiAktHmg13jLrFrQJhbANkByY40f9vxptC6pShcjLzEuHzx93ATo0c0XcSYs9W1RRbaDdu/pub?gid=1280822572&single=true&output=csv")
    # eval_index = range(2)
    # query_answer = query_answer.iloc[eval_index]
    
    question_col = "Question"
    channel_id_col = "GT: Retrieved channel id"
    question_list = list(query_answer[question_col])
    ground_type = "Document Retrieval"
    # make it as list as the length equals to question_list
    ground_type_list = [ground_type] * len(question_list)
    ground_channel_list = list(query_answer[channel_id_col])
    return question_list, ground_type_list, ground_channel_list

@pytest.mark.asyncio
async def test_chatbot_with_eval():
    restriction = "Use Document Retrieval to answer the question. Make sure to retrieve from bioimage.io channel"
    question = "What is a Bioimage Model Zoo community partner?" + restriction
    chat_history=[]
    profile = UserProfile(name="lulu", occupation="data scientist", background="machine learning and AI")
    m = QuestionWithHistory(question=question, chat_history=chat_history, user_profile=UserProfile.parse_obj(profile), channel_id=None)
    resp = await customer_service.handle(Message(content=m.json(), data=m , role="User"))
    resp_type =resp[0].data.steps[0].name
    assert resp_type == "Document Retrieval"
    channel_id = resp[0].data.steps[0].details["channel_id"]
    assert channel_id == "bioimageio"
    # Eval score
    ref_answer = "A BioImage Model Zoo community partner is an organization, company, research group, or software team that can consume and/or produce resources of the BioImage.IO model zoo. These partners continuously and openly contribute resources of their own, and they can participate in the decision-making process of the model specification. Additionally, they can show their logo in BioImage.IO, connect CI to automatically test new model compatibility with their software, and use other infrastructure features provided by BioImage.IO. The community partners can host their own Github repository for storing models and other relevant resources, which are then dynamically linked to the central repository of BioImage.IO. Each community partner is responsible for maintaining the resources that are relevant."
    eval_prompt = "Float between 0 and 5 representing the similarity score, where 5 means the same and 0 means not similar, how similar in meaning is the `llm_answer` to the `reference_answer`. It should be 0 if there is factual error detected! "
    
@pytest.mark.asyncio
async def test_chatbot(eval_questions):
    question_list, ground_type, ground_channel = eval_questions
    
    for question, resp_type_ref, channel_id_ref in zip(question_list, ground_type, ground_channel):
        chat_history=[]
        # restriction = "Use retrieval to answer the question."
        profile = UserProfile(name="lulu", occupation="data scientist", background="machine learning and AI")
        m = QuestionWithHistory(question=question, chat_history=chat_history, user_profile=UserProfile.parse_obj(profile), channel_id=None)
        resp = await customer_service.handle(Message(content=m.json(), data=m , role="User"))
        resp_type =resp[0].data.steps[0].name
        assert resp_type == resp_type_ref
        channel_id = resp[0].data.steps[0].details["channel_id"]
        assert channel_id == channel_id_ref

