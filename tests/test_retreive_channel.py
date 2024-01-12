import os
from bioimageio_chatbot.chatbot import create_customer_service, QuestionWithHistory, UserProfile
from schema_agents.schema import Message
import pandas as pd
import asyncio
import pytest

KNOWLEDGE_BASE_PATH = "./bioimageio-knowledge-base"
customer_service = create_customer_service(KNOWLEDGE_BASE_PATH)

# @pytest.mark.asyncio
async def test_chatbot_answer_channel(question, resp_type_ref, channel_id_ref):
    chat_history=[]
    # restriction = "Use retrieval to answer the question."
    
    profile = UserProfile(name="lulu", occupation="data scientist", background="machine learning and AI")
    m = QuestionWithHistory(question=question, chat_history=chat_history, user_profile=UserProfile.parse_obj(profile), channel_id=None)
    resp = await customer_service.handle(Message(content=m.json(), data=m , role="User"))
    resp_type =resp[0].data.steps[0].name
    channel_id = resp[0].data.steps[0].details["channel_id"]
    assert resp_type == resp_type_ref
    assert channel_id == channel_id_ref
    


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    eval_file = "Knowledge-Retrieval-Evaluation - Test_Wanlu.csv"
    query_answer = pd.read_csv(os.path.join(dir_path, eval_file))
    # eval_index = range(2)
    # query_answer = query_answer.iloc[eval_index]
    
    question_col = "Question"
    channel_id_col = "GT: Retrieved channel id"
    question_list = list(query_answer[question_col])
    ground_type = "Document Retrieval"
    ground_channel = list(query_answer[channel_id_col])
    
    tasks = [test_chatbot_answer_channel(question, resp_type_ref, channel_id_ref) for question, resp_type_ref, channel_id_ref in zip(question_list, ground_type, ground_channel)]

    async def run_tasks(tasks):
        await asyncio.gather(*tasks)
    # Run tasks concurrently
    asyncio.run(run_tasks(tasks))
    