import logging
import json
from api import call_openai_model_with_tools_ali as call_summarizer_llm
from utils import fix_and_parse_json

class ReflectionAgent:
    def __init__(self, question, data_name, logger):
        self.question = question
        self.logger = logger
        self.data_name = data_name

        self.S_prompt = (
" You are a rigorous reflection agent. Your task is to critically evaluate the entire problem-solving process of the core agent."
" You will be provided with the complete operational history and the proposed final answer."
" Your goal is to identify any potential errors, such as omissions of information, reasoning fallacies, or fabrication of facts."
" You must return the evaluation results in JSON format."
)

        self.prompt = """Please evaluate the credibility of the entire problem-solving process and the proposed answer based on the following information:
Operations performed by the core agent to solve a video understanding problem include:
{history}

The original video understanding question:
Question: {question}

The final answer proposed by the core agent:
Proposed Answer: {proposed_answer}

Evaluation Criteria:
- If the process and answer are credible and correct, set "credible"to true.
- If any errors are found, set "credible"to falseand provide a concise explanation stating what the issue is and why the proposed answer is incorrect.

Please respond strictly in the following JSON format:
{{
"credible": boolean, // true means the answer is credible, false means it is not
"comment": "Your concise explanation. This should be null if credible is true"
}}

Please return only the JSON object."""

    def _construct_messages(self, proposed_answer, history):
        # Convert history to a string format for the prompt
        history_str = "\n".join([json.dumps(h) for h in history])
        
        messages = [
            {"role": "system", "content": self.S_prompt},
            {"role": "user", "content": self.prompt.format(
                history=history_str, 
                question=self.question, 
                proposed_answer=proposed_answer
            )},
        ]
        return messages

    def run(self, proposed_answer: str, history) -> dict:
        """
        Run a single reflection step and return a structured assessment.
        """
        msgs = self._construct_messages(proposed_answer, history)

        response = call_summarizer_llm(
            messages=msgs,
            temperature=0.0,
        )

        if response and response.get("content"):
            self.logger.info(f"ReflectionAgent raw response: {response.get('content')}")
            assessment = fix_and_parse_json(response.get('content'), self.logger)
            if assessment and isinstance(assessment, dict) and 'credible' in assessment:
                self.logger.info(f"ReflectionAgent assessment: {assessment}")
                return assessment
        
        self.logger.warning("Failed to get a valid assessment from ReflectionAgent.")
        # Fallback: assume the answer is credible to avoid getting stuck
        return {"credible": True, "comment": "Fallback: Could not get a valid reflection."}