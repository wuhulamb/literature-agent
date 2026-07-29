from openai import OpenAI

from literature_agent.agents.base import BaseAgent
from literature_agent.models.document import Document, DocumentMetadata


_EXTRACT_SYSTEM_PROMPT = """You are a research paper metadata extractor. Given the first few pages of an academic paper, extract its bibliographic metadata.

Rules:
- title: The paper's full title. If unclear, return null.
- authors: List of author names as they appear. If none found, return empty list.
- year: The publication year as a 4-digit integer. If unclear, return null.
- journal: The journal or venue name. If unclear, return null.
- language: One of "en" (English), "zh" (Chinese), "ja" (Japanese). Default to "en" if not obvious.

Respond with ONLY valid raw JSON. No markdown, no code fences, no explanation. Do not wrap the response in ```json or any other tags. If a field cannot be determined from the text, set it to null (or empty list for authors)."""


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
                {"role": "system", "content": _EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": input_text},
            ],
            response_format=DocumentMetadata,
        )

        metadata = completion.choices[0].message.parsed
        if metadata is not None:
            document.metadata = metadata

        return document
