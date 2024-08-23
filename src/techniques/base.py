from techniques import PromptingStrategy
from typing import List


class BaseTechnique(PromptingStrategy):
    def get_initial_prompt(self, question: str) -> str:
        return f"Please solve the following problem:\n\n {question}"

    def get_follow_up_prompt(self, question: str, previous_response: str) -> None:
        return None

    def is_complete(self, question: str, responses: List[str]) -> bool:
        return len(responses) > 0

class StraightToAnswer(PromptingStrategy):
    def get_initial_prompt(self, question: str) -> str:
        return f"""
        Provide a direct and concise answer to the following question:
        
        {question}

        Only output the answer and nothing else
        """

    def get_follow_up_prompt(self, question: str, previous_response: str) -> None:
        return None

    def is_complete(self, question: str, responses: List[str]) -> bool:
        return len(responses) > 0