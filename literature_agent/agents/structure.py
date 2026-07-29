import json

from pydantic import BaseModel

from openai import OpenAI

from literature_agent.agents.base import BaseAgent
from literature_agent.models.document import Document, DocumentBlock, DocumentStructure, SectionNode


class BlockHeadingAnnotation(BaseModel):
    block_id: int
    is_heading: bool
    title: str | None = None
    level: int | None = None


class BatchHeadingResult(BaseModel):
    annotations: list[BlockHeadingAnnotation]


_HEADING_DETECTOR_PROMPT = """
## role

You are an academic document structure analyzer.

You are part of a multi-stage document parsing pipeline.

A previous agent has already extracted the document metadata, including the paper title.

The paper title has already been identified and stored separately. It is NOT part of your task.

Your responsibility is to identify structural section headings within the document body and infer their hierarchy.

Do not attempt to identify the paper title, author information, affiliations, abstracts, keywords, or other metadata. Focus only on headings that define the logical structure of the paper.

## Input

The input is a JSON object with the following fields:

### existing_headings

A list of structural headings that have already been confirmed in previous batches.

Each heading includes its id, title, and hierarchy level.

### blocks

The current batch of text blocks to analyze.

Each block contains an id and its extracted text.

## Task

Your task is to analyze each text block in the input batch and determine whether it represents or starts with a structural section heading.

For each block, you must output an annotation containing the following three fields:

1. **is_heading**
   A boolean flag indicating whether the block contains or starts with a structural section heading.

2. **title**
   The extracted text of the section heading.
   - If `is_heading` is `true`, return only the heading text.
   - If `is_heading` is `false`, return `null`.

3. **level**
   An integer representing the hierarchical level of the structural section heading.
   - **Level 1**: Top-level major sections (e.g., "1 Introduction", "Methods", "Discussion", "Conclusion")
   - **Level 2**: Subsections (e.g., "2.1 Background", "3.2 Model Specification")
   - **Level 3+**: Deeper nested sub-subsections (e.g., "3.1.1 Data Collection")
   - If `is_heading` is `false`, return `null`.

## Decision Rules

A valid structural section heading must satisfy ALL of the following:

1. It introduces a new structural section, and the following content belongs to that section until another heading appears.

2. It fits naturally into the document hierarchy and is neither the document title nor an exact duplicate of an existing heading.

3. Any standalone section number pattern such as "1", "1.1", "1.1.1", "第3章", "III." followed by title-like text should be treated as a structural heading even when it is embedded within a paragraph due to OCR line merging.

4. A structural heading may appear at the beginning or anywhere inside a block due to OCR or page merging. Ignore any residual body text before the heading.

5. If a numbered section heading (e.g. "2", "2.1", "3.3", "III.", "第3章") appears in a block and is followed by section content, treat it as a structural heading. Extract ONLY the heading text.

6. If a block contains both a heading and body text, return only the heading and ignore the surrounding body text.

7. Infer the hierarchy level from the heading numbering whenever possible (e.g. "2" → Level 1, "2.1" → Level 2, "2.1.1" → Level 3).

## Common Non-Headings

The following are NOT section headings:

- Paper title or any part of the paper title
- Author names, affiliations, or correspondence information
- Journal, publisher, or venue names (e.g. "Research Policy", "Nature", "Science")
- Abstract
- Keywords
- JEL codes or classification codes
- Article metadata (e.g. DOI, received/accepted/published dates)
- Page headers, page footers, or running headers
- Figure or table captions

## Common Headings

Common section headings include, but are not limited to:

- Numbered sections (e.g. "I. Introduction", "2 Methods", "3.1 Data")
- Introduction
- Background
- Related Work
- Methods or Methodology
- Results
- Discussion
- Conclusion
- References or Bibliography
- Acknowledgements
- Appendix
- Supplementary Material
- Data Availability Statement
- CRediT Authorship Contribution Statement
- Declaration of Competing Interest

## Examples

Input block:

4.1 模型与变量设定 为识别促进和阻碍中国与世界市场建立产业联系的动力……

Output:

is_heading = true
title = "4.1 模型与变量设定"
level = 2

Input block:

3.2 Model Specification We estimate the following regression...

Output:

is_heading = true
title = "3.2 Model Specification"
level = 2

Input block:

欧社群（瑞典—丹麦—芬兰）。 3.3   中国在中国—世界产业联系网络中的角色与地位演变 本文进一步从影响力……

Output:

is_heading = true
title = "3.3 中国在中国—世界产业联系网络中的角色与地位演变"
level = 2

## Output

Return ONLY a JSON object matching this schema:

{
  "annotations": [
    {
      "block_id": integer,
      "is_heading": boolean,
      "title": string or null,
      "level": integer or null
    }
  ]
}

Include ALL blocks from the input in the output, preserving order. Do not use markdown. Do not wrap the response in code fences."""  # fmt: skip


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
