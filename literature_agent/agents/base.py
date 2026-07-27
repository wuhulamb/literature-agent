from abc import ABC, abstractmethod

from pydantic import BaseModel


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    @abstractmethod
    def run(
        self,
        document: BaseModel,
    ) -> BaseModel:
        """Process document and return updated version.

        Args:
            document: The Document model to process.

        Returns:
            The same Document with relevant fields populated.
        """
        ...