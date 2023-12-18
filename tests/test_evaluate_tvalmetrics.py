import os
import numpy as np
from matplotlib import pyplot as plt
from bioimageio_chatbot.chatbot import create_customer_service, QuestionWithHistory, UserProfile
from schema_agents.schema import Message
from schema_agents.role import Role
from tvalmetrics import RagScoresCalculator
from itertools import cycle
import pandas as pd
import asyncio
import yaml
import pytest

KNOWLEDGE_BASE_PATH = "./bioimageio-knowledge-base"
# Dir of the current file
dir_path = os.path.dirname(os.path.realpath(__file__))
result_path = os.path.join(dir_path, "score_results")
# read Knowledge-Retrieval-Evaluation - Hoja 1 csv file
query_answer = pd.read_csv(os.path.join(dir_path, "Knowledge-Retrieval-Evaluation - Hoja 1.csv"))
score_result = []

def create_chatgpt():
    async def respond_direct(question: str, role: Role) -> str:
        """Generate a response to a question."""
        return await role.aask(question)
        
    chatgpt = Role(
        name="ChatGPT",
        model="gpt-3.5-turbo-1106",
        profile="Customer Service",
        goal="You are responsible for answering user questions, providing clarifications. Your overarching objective is to make the user experience both educational and enjoyable.",
        constraints=None,
        actions=[respond_direct],
    )
    return chatgpt
def load_query_answer():
    # read Knowledge-Retrieval-Evaluation - Hoja 1 csv file
    query_answer = pd.read_csv(os.path.join(dir_path, "Knowledge-Retrieval-Evaluation - Hoja 1.csv"))
    return query_answer
# @pytest.mark.asyncio
async def evaluate_retrieval():
    # customer_service = create_customer_service(KNOWLEDGE_BASE_PATH)
    # chat_history = []
    # BMZ_chatbot_answer_list = []
    # chatgpt_answer_list = []
    # # get query_answer from yaml file
    # for i in range(len(query_answer)):
    #     question = query_answer.iloc[i]["Question"]
    #     reference_answer = query_answer.iloc[i]["ChatGPT (3.5) Answer"]
    #     retrieved_context_list = query_answer.iloc[i]["Documentation"]
        
    #     # chatgpt answer
    #     chatgpt = create_chatgpt()
    #     chatgpt_answer = await chatgpt.handle(Message(content=question, role="User"))
    #     query_answer.loc[i, 'ChatGPT Direct Answer'] = chatgpt_answer[0].content
    #     chatgpt_answer_list.append(chatgpt_answer)
    #     # BioImage.IO Chatbot
    #     profile = UserProfile(name="", occupation="", background="")
    #     m = QuestionWithHistory(question=question, chat_history=chat_history, user_profile=UserProfile.parse_obj(profile), channel_id=None)
    #     llm_answer = await customer_service.handle(Message(content=m.json(), data=m , role="User"))
    #     # add a column to query_answer, content is llm_answer[0].content
    #     query_answer.loc[i, 'BioImage.IO Chatbot Answer'] = llm_answer[0].content
    #     BMZ_chatbot_answer_list.append(llm_answer)
        
    # # save query_answer to original csv file
    # query_answer.to_csv(os.path.join(dir_path, "Knowledge-Retrieval-Evaluation - Hoja 1.csv"))
    query_answer = load_query_answer()
    chatgpt_answer_list = list(query_answer['ChatGPT Direct Answer'])
    BMZ_chatbot_answer_list = list(query_answer['BioImage.IO Chatbot Answer'])
    
    question_list = list(query_answer['Question'])
    reference_answer_list = list(query_answer['ChatGPT (3.5) Answer'])
    chatgpt_score_calculator = RagScoresCalculator(
        model="gpt-4",
        answer_similarity_score=True,)
    # You only specify the inputs that are needed to calculate the specified scores.
    chatgpt_scores = chatgpt_score_calculator.score(question_list[0], reference_answer_list[0], chatgpt_answer_list[0], list([""]*(len(query_answer['Question'])-5)))
    chatgpt_answer_content_list = chatgpt_answer_list#[chatgpt_scores.llm_answer_list[i][0].content for i in range(len(chatgpt_scores.llm_answer_list))]
    
    # evaluate chatbot answer
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
    scores = score_calculator.score_batch(list(query_answer['Question']), list(query_answer['ChatGPT (3.5) Answer']), BMZ_chatbot_answer_list, list(query_answer['Documentation']))
    # print all elements in scores, scors is not a dict, is a Scores object
    print(f"answer_similarity_score: {scores.answer_similarity_score_list}, answer_consistency: {scores.answer_consistency_list}, retrieval_precision: {scores.retrieval_precision_list}, augmentation_precision: {scores.augmentation_precision_list}, augmentation_accuracy: {scores.augmentation_accuracy_list}")
    llm_answer_content_list = BMZ_chatbot_answer_list#[scores.llm_answer_list[i][0].content for i in range(len(scores.llm_answer_list))]
    # Convert Scores object to dictionary
    scores_dict = {
        "answer_similarity_score_list": scores.answer_similarity_score_list,
        "answer_consistency_list": scores.answer_consistency_list,
        "retrieval_precision_list": scores.retrieval_precision_list,
        "augmentation_precision_list": scores.augmentation_precision_list,
        "augmentation_accuracy_list": scores.augmentation_accuracy_list,
        "question_list": scores.question_list,
        "llm_answer_content_list": llm_answer_content_list,
        "retrieved_context_list": scores.retrieved_context_list_list,
        "chatgpt_scores": chatgpt_scores.answer_similarity_score_list,
        "chatgpt_answer_content_list": chatgpt_answer_content_list,
        }
    score_result.append(scores_dict)
    print(f"Scores: {scores_dict}")
    # save result to yaml file
    with open(os.path.join(result_path, "score_result.yaml"), 'w') as yaml_file:
        yaml.dump(score_result, yaml_file)
        

def visualize_score_result():
    # Visualize score result
    # Dir of the current file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    result_path = os.path.join(dir_path, "score_results")

    # Create an empty list to store the data
    data_list = []

    # Loop through files in result_path
    for file in os.listdir(result_path):
        # Get file path
        file_path = os.path.join(result_path, file)

        # Open file
        with open(file_path, 'r') as yaml_file:
            score_result = yaml.load(yaml_file, Loader=yaml.FullLoader)

            # Loop through score_result
            for i in range(len(score_result)):
                # Get score
                score = score_result[i]

                # Get score lists
                answer_similarity_score_list = score["answer_similarity_score_list"]
                answer_consistency_list = score["answer_consistency_list"]
                retrieval_precision_list = score["retrieval_precision_list"]
                augmentation_precision_list = score["augmentation_precision_list"]
                augmentation_accuracy_list = score["augmentation_accuracy_list"]

                # Create a dictionary with the data
                data = {
                    "File Name": file,
                    "Question Number": i + 1,
                    "Answer Similarity": answer_similarity_score_list[0],
                    "Answer Consistency": answer_consistency_list[0],
                    "Retrieval Precision": retrieval_precision_list[0],
                    "Augmentation Precision": augmentation_precision_list[0],
                    "Augmentation Accuracy": augmentation_accuracy_list[0]
                }
                # Append the data to the list
                data_list.append(data)

    # Create the DataFrame using the list of dictionaries
    df = pd.DataFrame(data_list)
    # save to csv file
    df.to_csv(os.path.join(result_path, "score_result.csv"))
    # Display the DataFrame
    print(df)
if __name__ == "__main__":
    asyncio.run(evaluate_retrieval())
    # visualize_score_result()