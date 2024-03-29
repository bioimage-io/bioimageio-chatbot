from pydantic import BaseModel, Field
from schema_agents.schema import Message
from typing import Any, Dict, List, Optional, Union
from schema_agents.role import Role

class EvaluationCriteria(BaseModel):
    relevance_and_accuracy: str = Field(
        default="Score 0-100: 0 means completely irrelevant, providing no useful information. "
                "100 means the answer is comprehensive, accurate, and closely matches the reference answer.",
        description="Assess how relevant and accurate the chatbot's answer is compared to the reference answer."
    )
    coverage_of_key_points: str = Field(
        default="Baseline >60: Answers covering the main points from the reference should score above 60, "
                "indicating they address the primary aspects of the question.",
        description="Evaluate whether the chatbot's answer includes the main points mentioned in the reference answer."
    )
    additional_information: str = Field(
        default="Variable Impact: Additional helpful information can increase the score. "
                "Irrelevant or unhelpful information should lead to a reduced score.",
        description="Assess the impact of additional information not present in the reference answer."
    )
    evaluation_guidelines: str = Field(
        default="Apply scoring criteria consistently and impartially. "
                "Provide justification for scores, especially for significant deviations from the baseline.",
        description="Guidelines for objective and transparent evaluation."
    )
    
class EvalInput(BaseModel):
    """Input for evaluating scores of LLM-based system."""
    question: str = Field(description="The question that was asked.")
    reference_answer: str = Field(description="The answer that was expected.")
    llm_answer: str = Field(description="The answer that was generated by the LLM-based system.")
    
class EvalScores(BaseModel):
    """Scores of evaluating llm answer."""
    criteria: EvaluationCriteria = Field(description="Criteria for evaluating the performance of the LLM-based system.")
    similarity_score: float = Field(description="Following the criteria, access the llm_answer. Float between 0 and 100 representing the similarity score. ")
    
def create_eval_agent():
    async def bot_answer_evaluate(req: EvalInput, role: Role) -> EvalScores:
        """Return the answer to the question."""
        response = await role.aask(req, EvalScores)
        return response
    
    eval_bot = Role(
        name="Thomas",
        profile="Evaluator",
        goal="Evaluate the performance of the LLM-based system.",
        constraints=None,
        actions=[bot_answer_evaluate],
        model="gpt-4-1106-preview"
    )
    return eval_bot

async def evaluate(question, reference_answer, llm_answer):
    eval_bot = create_eval_agent()
    eval_input = EvalInput(question=question, reference_answer=reference_answer, llm_answer=llm_answer)
    scores = await eval_bot.handle(Message(content=eval_input.model_dump_json(), data=eval_input, role="User"))
    similarity_score = scores[0].data.similarity_score
    return similarity_score
    