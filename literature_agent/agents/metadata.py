from openai import OpenAI

from literature_agent.agents.base import BaseAgent
from literature_agent.models.document import Document, DocumentMetadata
from literature_agent.prompts import load_prompt


class MetadataAgent(BaseAgent):
    """Extract bibliographic metadata from the first 2 pages of a document."""

    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def run(self, document: Document) -> Document:
        pages = document.content.pages or []
        first_two = pages[:2]
        if not first_two:
            return document

        input_text = "\n\n".join(p.text for p in first_two)

        completion = self._client.beta.chat.completions.parse(
            model="ecnu-max",
            messages=[
                {"role": "system", "content": load_prompt("metadata")},
                {"role": "user", "content": input_text},
            ],
            response_format=DocumentMetadata,
        )

        metadata = completion.choices[0].message.parsed
        if metadata is not None:
            document.metadata = metadata

        return document
