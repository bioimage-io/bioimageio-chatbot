import os
import numpy as np
from matplotlib import pyplot as plt
from bioimageio_chatbot.chatbot import create_customer_service, QuestionWithHistory, UserProfile
from pydantic import BaseModel, Field
from schema_agents.schema import Message
from typing import Any, Dict, List, Optional, Union
from schema_agents.role import Role
from tvalmetrics import RagScoresCalculator
from itertools import cycle
from bs4 import BeautifulSoup
import pandas as pd
import asyncio
import yaml


dir_path = os.path.dirname(os.path.realpath(__file__))
KNOWLEDGE_BASE_PATH = "./bioimageio-knowledge-base"

class EvalInput(BaseModel):
    """Input for evaluating scores of LLM-based system."""
    question: str = Field(description="The question that was asked.")
    reference_answer: str = Field(description="The answer that was expected.")
    llm_answer: str = Field(description="The answer that was generated by the LLM-based system.")
    retrieved_context_list: Optional[List[str]] = Field(description="Retrieved context used by the LLM-based system to make answer.")
    top_k_context_list: Optional[List[str]] = Field(description="Top k contexts that would be retrieved by the LLM-based system. There's an assumption that the retrieved context list is a subset of the top k context list.")
    
class ContextConsistency(BaseModel):
    """Scores of evaluating consistency between LLM answer and retrieved context."""
    main_point_list: List[str] = Field(description="List of main points of `llm_answer`. Each main point is a string, maximum summarize 10 points.")
    main_point_derived_from_context_list: List[bool] = Field(description="List of booleans representing whether each main point in `main_point_list` was derived from context in `retrieved_context_list`.")
    
class ContextScores(BaseModel):
    """Scores of evaluating retrieval based llm answer."""
    retrieval_precision: List[bool] = Field(description="Representing the retrieval precision score. Considering `question` and `retrieved_context_list`, determine whether the context is relevant for answering the question. If the context is relevant for answering the question, respond with true. If the context is not relevant for answering the question, respond with false.")
    augmentation_accuracy: List[bool] = Field(description="Representing the augmentation accuracy score. Considering `llm_answer` and `retrieved_context_list`, determine whether the answer contains information derived from the context. If the answer contains information derived from the context, respond with true. If the answer does not contain information derived from the context, respond with false. ")
    augmentation_consistency: ContextConsistency = Field(description="Whether there is information in the `llm_answer` that does not come from the context. Summarize main points in `llm_answer`, determine whether the statement in main points can be derived from the context. If the statement can be derived from the context response with true. Otherwise response with false.")

class EvalScores(BaseModel):
    """Scores of evaluating llm answer."""
    similarity_score: float = Field(description="Float between 0 and 5 representing the similarity score, where 5 means the same and 0 means not similar, how similar in meaning is the `llm_answer` to the `reference_answer`. It should be 0 if there is factual error detected! ")
    context_scores: Optional[ContextScores] = Field(description="Scores of evaluating retrieval based llm answer.")
    
def extract_original_content(input_content):
    soup = BeautifulSoup(input_content, 'html.parser')

    # Check if the content is a table
    table = soup.find('table')
    if table:
        # Extract content from the table
        rows = table.find_all('tr')[1:]
        original_contents = []

        for row in rows:
            # Find all the cells in the row
            cells = row.find_all('td')

            # Check if there are at least two cells in the row
            if len(cells) >= 2:
                # Extract content from the second cell (index 1)
                content = cells[1].get_text(strip=True)
                original_contents.append(content)
    else:
        # Check if the content is in source code format
        code = soup.find('code')
        if code:
            # Extract content from the source code
            original_contents = [code.get_text(strip=True)]
        else:
            # Content format not recognized
            original_contents = None

    return original_contents

def load_query_answer(eval_file):
    # read Knowledge-Retrieval-Evaluation - Hoja 1 csv file
    query_answer = pd.read_csv(os.path.join(dir_path, eval_file))
    return query_answer

def create_gpt(model="gpt-3.5-turbo-1106"):
    async def respond_direct(question: str, role: Role) -> str:
        """Generate a response to a question."""
        return await role.aask(question)
        
    chatgpt = Role(
        name="GPT",
        model=model,
        profile="Customer Service",
        goal="You are responsible for answering user questions, providing clarifications. Your overarching objective is to make the user experience both educational and enjoyable.",
        constraints=None,
        actions=[respond_direct],
    )
    return chatgpt

async def get_answers(excel_file, eval_index=None):
    query_answer = pd.read_csv(excel_file)
    customer_service = create_customer_service(KNOWLEDGE_BASE_PATH)
    chatgpt = create_gpt()
    chat_history = []
    
    async def process_question(i):
        nonlocal query_answer, chat_history
        print(f"\n==================\nProcessing {i}th question...")
        question = query_answer.iloc[i]["Question"]
        
        # gpt-3.5-turbo answer
        chatgpt_answer = await chatgpt.handle(Message(content=question, role="User"))
        query_answer.loc[i, 'GPT-3.5-tubor Answer  (Without Context)'] = chatgpt_answer[0].content
        
        # BMZ chatbot
        event_bus = customer_service.get_event_bus()
        event_bus.register_default_events()
        profile = UserProfile(name="", occupation="", background="")
        m = QuestionWithHistory(question=question, chat_history=chat_history, user_profile=UserProfile.parse_obj(profile), channel_id=None)
        llm_answer = await customer_service.handle(Message(content=m.json(), data=m, role="User"))
        
        # separate llm_answer into answer and reference using "<details><summary>References</summary>" as separator
        llm_answer_list = llm_answer[0].content.split("<details><summary>")
        
        # add a column to query_answer, content is llm_answer[0].content
        query_answer.loc[i, 'BMZ Chatbot Answer'] = llm_answer_list[0]
        if len(llm_answer_list) > 1:
            query_answer.loc[i, 'BMZ Chatbot Answer References'] = llm_answer_list[1]
        else:
            query_answer.loc[i, 'BMZ Chatbot Answer References'] = "NA"
        
        print(f"Finish processing {i}th question!")

    if eval_index is None:
        eval_index = range(len(query_answer))

    # Create a list of tasks
    tasks = [process_question(i) for i in eval_index]

    # Run tasks concurrently
    await asyncio.gather(*tasks)
    # for task in tasks:
    #     await task
    # Save the updated DataFrame to the CSV file after all questions are processed
    query_answer.to_csv(os.path.join(dir_path, excel_file))

    print(f"Update {excel_file} successfully!")


async def get_ground_truth(excel_file, eval_index=None):
    query_answer = pd.read_csv(excel_file)
    prompt_chatgpt = "Here is the contexts I found from the documentation: ```\n{contextx}\n```\nNow based on the given context, please answer the question: ```\n{question}\n```."
    chatgpt = create_gpt(model="gpt-4-1106-preview")
    event_bus = chatgpt.get_event_bus()
    event_bus.register_default_events()
    if eval_index is None:
        eval_index = range(len(query_answer))
    for i in eval_index:
        prompt = prompt_chatgpt.format(contextx=query_answer.iloc[i]["Documentation"], question=query_answer.iloc[i]["Question"])
        ground_truth_answer = await chatgpt.handle(Message(content=prompt, role="User"))
        # query_answer.loc[i, 'GPT-3.5-turbo Answer (With Context)- GT'] = ground_truth_answer[0].content
        query_answer.loc[i, 'GPT-4-turbo Answer (With Context)- GT'] = ground_truth_answer[0].content
        # save query_answer to original csv file
        query_answer.to_csv(os.path.join(dir_path, excel_file))
    
async def start_evaluate(eval_file, eval_index=None):
    async def bot_answer_evaluate(req: EvalInput, role: Role) -> EvalScores:
        """Return the answer to the question."""
        response = await role.aask(req, EvalScores)
        return response
    
    evalBot = Role(
            name="Thomas",
            profile="Evaluator",
            goal="Evaluate the performance of the LLM-based system.",
            constraints=None,
            actions=[bot_answer_evaluate],
            model="gpt-4"
        )
    event_bus = evalBot.get_event_bus()
    event_bus.register_default_events()
    
    query_answer = load_query_answer(eval_file)
    question_list = list(query_answer['Question'])
    ground_truth_answer_list = list(query_answer['GPT-4-turbo Answer (With Context)- GT'])
    chatgpt_answer_list = list(query_answer['GPT-3.5-tubor Answer  (Without Context)'])
    gpt4_direct_answer_list = list(query_answer['GPT-4 Answer'])
    BMZ_chatbot_answer_list = list(query_answer['BMZ Chatbot Answer'])
    
    if eval_index is None:
        eval_index = range(len(question_list))
    for i in eval_index:#len(question_list)):
        print(f"Evaluating {i}th question...")
        question = question_list[i]
        reference_answer = ground_truth_answer_list[i]
        gpt_direct_answer = chatgpt_answer_list[i]
        gpt4_direct_answer = gpt4_direct_answer_list[i]
        chatbot_answer = BMZ_chatbot_answer_list[i]
        reference_table= query_answer.iloc[i]["BMZ Chatbot Answer References"]
        if pd.isna(reference_table) or reference_table == "":
            retrieved_context_list = []
        else:
            retrieved_context_list = extract_original_content(reference_table)
        # gpt3direct 
        eval_input_gpt3_direct = EvalInput(question=question, reference_answer=reference_answer, llm_answer=gpt_direct_answer)
        scores_gpt3_direct = await evalBot.handle(Message(content= eval_input_gpt3_direct.json(), data= eval_input_gpt3_direct, role="User"))
        # gpt4direct
        eval_input_gpt4_direct = EvalInput(question=question, reference_answer=reference_answer, llm_answer=gpt4_direct_answer)
        scores_gpt4_direct = await evalBot.handle(Message(content= eval_input_gpt4_direct.json(), data= eval_input_gpt4_direct, role="User"))
        
        # chatbot answer
        eval_input_chatbot = EvalInput(question=question, reference_answer=reference_answer, llm_answer=chatbot_answer, retrieved_context_list=retrieved_context_list[1:])
        try:
            scores_chatbot = await evalBot.handle(Message(content= eval_input_chatbot.json(), data= eval_input_chatbot, role="User"))
        
            SimilaryScore = scores_chatbot[0].data.similarity_score
            RetrievalPrecision = sum(scores_chatbot[0].data.context_scores.retrieval_precision) / len(scores_chatbot[0].data.context_scores.retrieval_precision)
            AugmentationAccuracy = sum(scores_chatbot[0].data.context_scores.augmentation_accuracy) / len(scores_chatbot[0].data.context_scores.augmentation_accuracy)
            AugmentationPrecision = sum(scores_chatbot[0].data.context_scores.augmentation_accuracy) / len(scores_chatbot[0].data.context_scores.augmentation_accuracy)
            AugmentationConsistency = sum(scores_chatbot[0].data.context_scores.augmentation_consistency.main_point_derived_from_context_list) / len(scores_chatbot[0].data.context_scores.augmentation_consistency.main_point_derived_from_context_list)
        # if there is an error, set all scores to 'NA'
        except Exception as e:
            print(e)
            # if Error code is 400, it means the question is too long, so we need to split the question into two parts
            # if e.code == 400:
                
            try:
                too_long_context = retrieved_context_list[-1][:15000]
                # replace the last context with the first part of the too long context
                retrieved_context_list[-1] = too_long_context
                eval_input_chatbot = EvalInput(question=question, reference_answer=reference_answer, llm_answer=chatbot_answer, retrieved_context_list=retrieved_context_list[1:])
                scores_chatbot = await evalBot.handle(Message(content= eval_input_chatbot.json(), data= eval_input_chatbot, role="User"))
    
                SimilaryScore = scores_chatbot[0].data.similarity_score
                RetrievalPrecision = sum(scores_chatbot[0].data.context_scores.retrieval_precision) / len(scores_chatbot[0].data.context_scores.retrieval_precision)
                AugmentationAccuracy = sum(scores_chatbot[0].data.context_scores.augmentation_accuracy) / len(scores_chatbot[0].data.context_scores.augmentation_accuracy)
                AugmentationPrecision = sum(scores_chatbot[0].data.context_scores.augmentation_accuracy) / len(scores_chatbot[0].data.context_scores.augmentation_accuracy)
                AugmentationConsistency = sum(scores_chatbot[0].data.context_scores.augmentation_consistency.main_point_derived_from_context_list) / len(scores_chatbot[0].data.context_scores.augmentation_consistency.main_point_derived_from_context_list)
            except Exception as e:
                print(e)
                SimilaryScore = 'NA'
                RetrievalPrecision = 'NA'
                AugmentationAccuracy = 'NA'
                AugmentationPrecision = 'NA'
                AugmentationConsistency = 'NA'
            
        # save scores to a new dataframe
        query_answer.loc[i, 'GPT-3.5-tubor Answer (Without Context)- Answer Similarity Score'] = scores_gpt3_direct[0].data.similarity_score
        query_answer.loc[i, 'GPT4 Direct Answer - Answer Similarity Score'] = scores_gpt4_direct[0].data.similarity_score
        query_answer.loc[i, 'BMZ Chatbot Answer - Similarity Score'] = SimilaryScore
        query_answer.loc[i, 'BMZ Chatbot Answer - Retrieval Precision'] = RetrievalPrecision
        query_answer.loc[i, 'BMZ Chatbot Answer - Augmentation Accuracy'] = AugmentationAccuracy
        query_answer.loc[i, 'BMZ Chatbot Answer - Augmentation Precision'] = AugmentationPrecision
        query_answer.loc[i, 'BMZ Chatbot Answer - Augmentation Consistency'] = AugmentationConsistency
        print(f"Finish evaluating {i}th question!")
        # save query_answer to a new csv file
        query_answer.to_csv(os.path.join(dir_path, eval_file))
    # save query_answer to original csv file
    # query_answer.to_csv(os.path.join(dir_path, eval_file))
import asyncio

async def start_evaluate_paral(eval_file, eval_index=None):
    async def bot_answer_evaluate(req: EvalInput, role: Role) -> EvalScores:
        """Return the answer to the question."""
        response = await role.aask(req, EvalScores)
        return response
    
    evalBot = Role(
        name="Thomas",
        profile="Evaluator",
        goal="Evaluate the performance of the LLM-based system.",
        constraints=None,
        actions=[bot_answer_evaluate],
        model="gpt-4"
    )
    event_bus = evalBot.get_event_bus()
    event_bus.register_default_events()
    
    query_answer = load_query_answer(eval_file)
    question_list = list(query_answer['Question'])
    ground_truth_answer_list = list(query_answer['GPT-4-turbo Answer (With Context)- GT'])
    chatgpt_answer_list = list(query_answer['GPT-3.5-tubor Answer (Without Context)'])
    gpt4_direct_answer_list = list(query_answer['GPT-4 Answer'])
    BMZ_chatbot_answer_list = list(query_answer['BMZ Chatbot Answer'])
    
    if eval_index is None:
        eval_index = range(len(question_list))
    
    async def evaluate_question(i):
        print(f"Evaluating {i}th question...")
        question = question_list[i]
        reference_answer = ground_truth_answer_list[i]
        gpt_direct_answer = chatgpt_answer_list[i]
        gpt4_direct_answer = gpt4_direct_answer_list[i]
        chatbot_answer = BMZ_chatbot_answer_list[i]
        reference_table = query_answer.iloc[i]["BMZ Chatbot Answer References"]
        if pd.isna(reference_table) or reference_table == "":
            retrieved_context_list = [""]
        else:
            retrieved_context_list = extract_original_content(reference_table)
        
        # gpt3direct
        # eval_input_gpt3_direct = EvalInput(question=question, reference_answer=reference_answer, llm_answer=gpt_direct_answer)
        # # gpt4direct
        # eval_input_gpt4_direct = EvalInput(question=question, reference_answer=reference_answer, llm_answer=gpt4_direct_answer)
        
        # chatbot answer
        eval_input_chatbot = EvalInput(question=question, reference_answer=reference_answer, llm_answer=chatbot_answer, retrieved_context_list=retrieved_context_list)
        
        try:
            tasks = [
                # evalBot.handle(Message(content=eval_input_gpt3_direct.json(), data=eval_input_gpt3_direct, role="User")),
                # evalBot.handle(Message(content=eval_input_gpt4_direct.json(), data=eval_input_gpt4_direct, role="User")),
                evalBot.handle(Message(content=eval_input_chatbot.json(), data=eval_input_chatbot, role="User"))
            ]
            results = await asyncio.gather(*tasks)
            
            # scores_gpt3_direct, scores_gpt4_direct, scores_chatbot = results
            scores_chatbot = results[0]
            
            SimilaryScore = scores_chatbot[0].data.similarity_score
            RetrievalPrecision = sum(scores_chatbot[0].data.context_scores.retrieval_precision) / len(scores_chatbot[0].data.context_scores.retrieval_precision)
            AugmentationAccuracy = sum(scores_chatbot[0].data.context_scores.augmentation_accuracy) / len(scores_chatbot[0].data.context_scores.augmentation_accuracy)
            AugmentationPrecision = sum(scores_chatbot[0].data.context_scores.augmentation_accuracy) / len(scores_chatbot[0].data.context_scores.augmentation_accuracy)
            AugmentationConsistency = sum(scores_chatbot[0].data.context_scores.augmentation_consistency.main_point_derived_from_context_list) / len(scores_chatbot[0].data.context_scores.augmentation_consistency.main_point_derived_from_context_list)
            
            # query_answer.loc[i, 'GPT-3.5-tubor Answer (Without Context)- Answer Similarity Score'] = scores_gpt3_direct[0].data.similarity_score
            # query_answer.loc[i, 'GPT4 Direct Answer - Answer Similarity Score'] = scores_gpt4_direct[0].data.similarity_score
            query_answer.loc[i, 'BMZ Chatbot Answer - Similarity Score'] = SimilaryScore
            query_answer.loc[i, 'BMZ Chatbot Answer - Retrieval Precision'] = RetrievalPrecision
            query_answer.loc[i, 'BMZ Chatbot Answer - Augmentation Accuracy'] = AugmentationAccuracy
            query_answer.loc[i, 'BMZ Chatbot Answer - Augmentation Precision'] = AugmentationPrecision
            query_answer.loc[i, 'BMZ Chatbot Answer - Augmentation Consistency'] = AugmentationConsistency     
            
        except Exception as e:
            tasks = [
                # evalBot.handle(Message(content=eval_input_gpt3_direct.json(), data=eval_input_gpt3_direct, role="User")),
                # evalBot.handle(Message(content=eval_input_gpt4_direct.json(), data=eval_input_gpt4_direct, role="User")),
                evalBot.handle(Message(content=eval_input_chatbot.json(), data=eval_input_chatbot, role="User"))
            ]
            results = await asyncio.gather(*tasks)
            
            # scores_gpt3_direct, scores_gpt4_direct, scores_chatbot = results
            scores_chatbot = results[0]
            
            SimilaryScore = scores_chatbot[0].data.similarity_score
            # query_answer.loc[i, 'GPT-3.5-tubor Answer (Without Context)- Answer Similarity Score'] = scores_gpt3_direct[0].data.similarity_score
            # query_answer.loc[i, 'GPT4 Direct Answer - Answer Similarity Score'] = scores_gpt4_direct[0].data.similarity_score
            query_answer.loc[i, 'BMZ Chatbot Answer - Similarity Score'] = SimilaryScore

    # Create a list of tasks
    tasks = [evaluate_question(i) for i in eval_index]

    # Run tasks concurrently
    await asyncio.gather(*tasks)
    query_answer.to_csv(os.path.join(dir_path, eval_file))  # Save the updated DataFrame to the CSV file
 
async def get_answer_and_evaluate(excel, eval_index=None):
    # run get_answers
    await get_answers(excel, eval_index)
    # run start_evaluate
    await start_evaluate(excel, eval_index)
    print("Finish evaluating!")
    
if __name__ == "__main__":
    # file = os.path.join(dir_path, "Knowledge-Retrieval-Evaluation - Hoja 1-2.csv")
    file_with_gt = os.path.join(dir_path, "Knowledge-Retrieval-Evaluation - low-scores.csv")
   
    # load query_answer
    query_answer = load_query_answer(file_with_gt)
    # find the index of questions which 'BMZ Chatbot Answer - Similarity Score' is lower than 3
    # eval_index = query_answer[query_answer['BMZ Chatbot Answer - Similarity Score'] < 3].index
    # # find the index of questions which 'BMZ Chatbot Answer - Similarity Score' is NA
    # eval_index = query_answer[query_answer['BMZ Chatbot Answer - Similarity Score'].isna()].index
    # eval_index=range(2,len(query_answer))
    # asyncio.run(get_ground_truth(file_with_gt, eval_index))
    # asyncio.run(get_answers(file_with_gt))
    # asyncio.run(start_evaluate(file_with_gt))
    # asyncio.run(get_answer_and_evaluate(file_with_gt,eval_index))
    asyncio.run(start_evaluate_paral(file_with_gt))