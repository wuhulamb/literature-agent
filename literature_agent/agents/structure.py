import json

from pydantic import BaseModel

from openai import OpenAI

from literature_agent.agents.base import BaseAgent
from literature_agent.models.document import Document, DocumentBlock, DocumentStructure, SectionNode
from literature_agent.prompts import load_prompt


class BlockHeadingAnnotation(BaseModel):
    block_id: int
    is_heading: bool
    title: str | None = None
    level: int | None = None


class BatchHeadingResult(BaseModel):
    annotations: list[BlockHeadingAnnotation]


_HEADING_DETECTOR_PROMPT = load_prompt("structure_heading")


class StructureBuilder:
    """Build a section tree from a stream of heading annotations.

    Maintains a stack of open sections. On each new heading:
      - pop sections whose level >= incoming level
      - determine parent from the new stack top
      - push the new section
    Non-heading blocks are appended to the current (innermost) section's block_ids.
    """

    def __init__(self) -> None:
        self.section_stack: list[SectionNode] = []
        self._all_nodes: dict[int, SectionNode] = {}
        self.next_id = 0

    def process_heading(self, block_id: int, title: str, level: int) -> SectionNode:
        """Register a heading and return the newly created SectionNode."""
        while self.section_stack and self.section_stack[-1].level >= level:
            self.section_stack.pop()

        parent_id = self.section_stack[-1].id if self.section_stack else None

        node = SectionNode(
            id=self.next_id,
            title=title,
            level=level,
            parent_id=parent_id,
            block_ids=[block_id],
        )
        self.next_id += 1

        if parent_id is not None:
            self.section_stack[-1].children_ids.append(node.id)

        self._all_nodes[node.id] = node
        self.section_stack.append(node)
        return node

    def process_non_heading(self, block_id: int) -> None:
        """Assign a non-heading block to the current innermost section."""
        if self.section_stack:
            self.section_stack[-1].block_ids.append(block_id)

    @property
    def nodes(self) -> dict[int, SectionNode]:
        return self._all_nodes


class StructureAgent(BaseAgent):
    """Extract section hierarchy from document blocks.

    Two-phase approach:
      1. HeadingDetector (LLM): classify each block as heading or non-heading.
      2. StructureBuilder (Python): build section tree via stack operations.
    """

    def __init__(self, client: OpenAI, max_chars: int = 6000) -> None:
        self._client = client
        self._max_chars = max_chars

    def run(self, document: Document) -> Document:
        blocks_list = sorted(document.blocks.values(), key=lambda b: b.id)
        builder = StructureBuilder()

        # Use metadata title as root heading (level=0) so section_stack is never empty
        if document.metadata.title:
            builder.process_heading(-1, document.metadata.title, 0)

        batches: list[list[DocumentBlock]] = []
        current_batch: list[DocumentBlock] = []
        current_chars = 0

        for block in blocks_list:
            block_len = len(block.text)
            if current_chars + block_len > self._max_chars and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_chars = 0
            current_batch.append(block)
            current_chars += block_len
        if current_batch:
            batches.append(current_batch)

        existing_titles: set[str] = set()
        if document.metadata.title:
            existing_titles.add(document.metadata.title.lower().strip())

        for batch in batches:

            existing = [
                {"id": n.id, "title": n.title, "level": n.level}
                for n in builder.nodes.values()
            ]
            print(existing)

            payload = {
                "existing_headings": existing,
                "blocks": [{"id": b.id, "text": b.text} for b in batch],
            }

            batch_input = json.dumps(payload, ensure_ascii=False)

            completion = self._client.beta.chat.completions.parse(
                model="ecnu-max",
                messages=[
                    {"role": "system", "content": _HEADING_DETECTOR_PROMPT},
                    {
                        "role": "user",
                        "content": batch_input,
                    },
                ],
                response_format=BatchHeadingResult,
            )

            result = completion.choices[0].message.parsed

            # Map LLM annotations by block_id
            ann_by_id = {}
            if result is not None:
                for ann in result.annotations:
                    ann_by_id[ann.block_id] = ann

            # Iterate over input batch order, fill missing annotations with default
            batch_block_ids = [b.id for b in batch]
            for block_id in batch_block_ids:
                ann = ann_by_id.get(block_id)
                if ann is None:
                    ann = BlockHeadingAnnotation(block_id=block_id, is_heading=False)

                # Persist heading annotation back to the block
                block = document.blocks.get(block_id)
                if block is not None:
                    block.is_heading = ann.is_heading
                    block.heading_title = ann.title
                    block.heading_level = ann.level

                if ann.is_heading and ann.title and ann.level is not None:
                    title_lower = ann.title.lower().strip()
                    if title_lower in existing_titles:
                        builder.process_non_heading(ann.block_id)
                    else:
                        builder.process_heading(ann.block_id, ann.title, ann.level)
                        existing_titles.add(title_lower)
                else:
                    builder.process_non_heading(ann.block_id)

        document.structure = DocumentStructure(nodes=builder.nodes)
        return document
