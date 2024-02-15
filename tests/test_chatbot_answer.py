import os
from bioimageio_chatbot.chatbot import create_customer_service, get_builtin_extensions, QuestionWithHistory, UserProfile
from bioimageio_chatbot.evaluation import evaluate
from schema_agents.schema import Message
import json
import pandas as pd
import asyncio
import pytest

KNOWLEDGE_BASE_PATH = "./bioimageio-knowledge-base"
builtin_extensions = get_builtin_extensions()
extensions = [{key:value for key, value in ext.dict().items() if key in ["name", "description"]} for ext in builtin_extensions]
customer_service = create_customer_service(builtin_extensions)

dir_path = os.path.dirname(os.path.realpath(__file__))

@pytest.fixture
def eval_questions():
    
    eval_file = os.path.join(dir_path, "Minimal-Eval-Test-20240111.csv")
    if os.path.exists(eval_file):
        query_answer = pd.read_csv(eval_file)
    else:
        query_answer = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vTVgE2_eqBiAktHmg13jLrFrQJhbANkByY40f9vxptC6pShcjLzEuHzx93ATo0c0XcSYs9W1RRbaDdu/pub?gid=1280822572&single=true&output=csv")
    eval_index = range(1,10)
    query_answer = query_answer.iloc[eval_index]
    
    question_col = "Question"
    channel_id_col = "GT: Retrieved channel id"
    question_list = list(query_answer[question_col])
    reference_answer_list = list(query_answer["GPT-4-turbo Answer (With Context)- GT"])
    # ground_type = "Document Retrieval"
    # make it as list as the length equals to question_list
    # ground_type_list = [ground_type] * len(question_list)
    channel_id_list_gt = list(query_answer[channel_id_col])
    return question_list, reference_answer_list, channel_id_list_gt


async def validate_chatbot_answer(question, reference_answer, use_tools_gt, channel_id_gt, relevance_gt, similary_score_gt):
    chat_history=[]
    profile = UserProfile(name="", occupation="", background="")
    
    m = QuestionWithHistory(question=question, chat_history=chat_history, user_profile=UserProfile.parse_obj(profile), chatbot_extensions=extensions)
    resp = await customer_service.handle(Message(content=m.json(), data=m , role="User"))
    # use_tools =resp[0].data.steps[0].details["use_tools"]
    # assert use_tools == use_tools_gt
    # execute_tool = resp[0].data.steps[1].name
    # # get the string after 'Execute: '
    # channel_id = execute_tool.split(": ")[1]
    # assert channel_id == channel_id_gt+"(docs)"
    
    # eval score
    # relevance = resp[0].data.steps[-1].details["relevant"]
    # assert relevance == relevance_gt
    chatbot_answer = resp[0].data.steps[-1].details['details'][0]['response']
    similary_score = await evaluate(question, reference_answer, chatbot_answer)
    assert similary_score >= similary_score_gt

   
@pytest.mark.asyncio    
async def test_chatbot1(eval_questions):
    
#     await validate_chatbot_answer(
#         question="What is deepImageJ?",
#         reference_answer="DeepImageJ is a user-friendly plugin designed to facilitate the utilization of pre-trained neural networks within ImageJ and Fiji. It serves as a bridge between developers of deep-learning models and end-users in life-science applications, promoting the sharing of trained models across research groups. DeepImageJ is particularly valuable in various imaging domains and does not necessitate deep learning expertise or programming skills.",
#         use_tools_gt=True,
#         channel_id_gt="deepimagej(docs)",
#         relevance_gt=True,
#         similary_score_gt=4.0
#     )
    
#     await validate_chatbot_answer(
#         question="What is a Bioimage Model Zoo community partner?",
#         reference_answer="A BioImage Model Zoo community partner is an organization, company, research group, or software team that can consume and/or produce resources of the BioImage.IO model zoo. These partners continuously and openly contribute resources of their own, and they can participate in the decision-making process of the model specification. Additionally, they can show their logo in BioImage.IO, connect CI to automatically test new model compatibility with their software, and use other infrastructure features provided by BioImage.IO. The community partners can host their own Github repository for storing models and other relevant resources, which are then dynamically linked to the central repository of BioImage.IO. Each community partner is responsible for maintaining the resources that are relevant.",
#         use_tools_gt=True,
#         channel_id_gt="bioimage.io(docs)",
#         relevance_gt=True,
#         similary_score_gt=4.0
#     )
    
    questions, reference_answers, channel_id_list_gt = eval_questions
    for question, reference_answer, channel_id_gt in zip(questions, reference_answers, channel_id_list_gt):
        await validate_chatbot_answer(
            question=question,
            reference_answer=reference_answer,
            use_tools_gt=True,
            channel_id_gt=channel_id_gt,
            relevance_gt=True,
            similary_score_gt=80
        )