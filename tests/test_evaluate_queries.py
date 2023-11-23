import os
from bioimageio_chatbot.chatbot import create_customer_service, QuestionWithHistory, UserProfile
from schema_agents.schema import Message
from tvalmetrics import RagScoresCalculator
import asyncio
import yaml
import pytest

KNOWLEDGE_BASE_PATH = "./bioimageio-knowledge-base"
# dir of current file
dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, "query_answer.yaml"), 'r') as yaml_file:
    query_answer = yaml.load(yaml_file, Loader=yaml.FullLoader)
# @pytest.mark.asyncio
async def evaluate_retrieval():

    customer_service = create_customer_service(KNOWLEDGE_BASE_PATH)
    chat_history = []
    # get query_answer from yaml file
    for i in range(len(query_answer)):
        question = query_answer[i]["question"]
        reference_answer = query_answer[i]["reference_answer"]
        retrieved_context_list = query_answer[i]["retrieved_context_list"]
        
        profile = UserProfile(name="lulu", occupation="data scientist", background="machine learning and AI")
        m = QuestionWithHistory(question=question, chat_history=chat_history, user_profile=UserProfile.parse_obj(profile), channel_id=None)
        llm_answer = await customer_service.handle(Message(content=m.json(), data=m , role="User"))
        # assert resp

        llm_evaluator = "gpt-4-1106-preview"
        score_calculator = RagScoresCalculator(
            model=llm_evaluator,
            retrieval_precision=True,
            augmentation_precision=True,
            augmentation_accuracy=True,
            answer_consistency=True,
            answer_similarity_score=True,
        )

        # You only specify the inputs that are needed to calculate the specified scores.
        scores = score_calculator.score(
        question, reference_answer, llm_answer, retrieved_context_list)
        # print all elements in scores, scors is not a dict, is a Scores object
        print(f"answer_similarity_score: {scores.answer_similarity_score_list}, answer_consistency: {scores.answer_consistency_list}, retrieval_precision: {scores.retrieval_precision_list}, augmentation_precision: {scores.augmentation_precision_list}, augmentation_accuracy: {scores.augmentation_accuracy_list}")

if __name__ == "__main__":
    asyncio.run(evaluate_retrieval())