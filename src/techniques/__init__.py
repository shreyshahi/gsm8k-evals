from abc import ABC, abstractmethod
from typing import List, Optional

class PromptingStrategy(ABC):
    @abstractmethod
    def get_initial_prompt(self, question: str) -> str:
        """
        Generate the initial prompt for a given question.

        Args:
            question (str): The input question.

        Returns:
            str: The initial prompt to be sent to the model.
        """
        pass

    @abstractmethod
    def get_follow_up_prompt(self, question: str, previous_response: str) -> Optional[str]:
        """
        Generate a follow-up prompt based on the previous response.

        Args:
            question (str): The original input question.
            previous_response (str): The model's previous response.

        Returns:
            Optional[str]: The follow-up prompt if needed, or None if no further prompting is required.
        """
        pass

    @abstractmethod
    def is_complete(self, question: str, responses: List[str]) -> bool:
        """
        Determine if the prompting process is complete.

        Args:
            question (str): The original input question.
            responses (List[str]): A list of all responses received so far.

        Returns:
            bool: True if the prompting process is complete, False otherwise.
        """
        pass
