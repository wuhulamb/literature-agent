from openai import OpenAI

from literature_agent.agents.base import BaseAgent
from literature_agent.models.document import Document, DocumentBlock, DocumentSummary


_SUMMARY_SYSTEM_PROMPT = """You are an expert academic paper summarizer.

Output language:
- Simplified Chinese.

Writing style:
- Accurate.
- Concise.
- Objective.
- Academic.

Terminology:
- Keep technical terms, model names, benchmark names, dataset names,
  mathematical symbols, and abbreviations in their original language
  whenever appropriate.

Never fabricate information."""


_SECTION_SUMMARY_PROMPT = """Summarize the following section of a research paper.

Section title:
{title}

Write a concise summary in 2-4 sentences.

Text:
{text}"""


_PARENT_SUMMARY_PROMPT = """Synthesize the following subsection summaries into a concise summary of the parent section.

Section title:
{title}

Sub-section summaries:
{sub_summaries}

Additional content:
{additional_text}"""


_DOCUMENT_SUMMARY_PROMPT = """Write a concise summary of the entire paper in 4-6 sentences.

Section summaries:
{section_summaries}"""


_ONE_SENTENCE_PROMPT = """Summarize the paper in one sentence.

Paper summary:
{document_summary}"""


class SummaryAgent(BaseAgent):
    """Generate hierarchical summaries bottom-up along the section tree."""

    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def run(self, document: Document) -> Document:
        structure = document.structure
        if structure is None or not structure.nodes:
            return document

        nodes = structure.nodes
        section_summaries: dict[int, str] = {}

        # Post-order traversal: children before parent
        def _summarize(node_id: int) -> str:
            node = nodes[node_id]

            if node.children_ids:
                child_summaries = []
                for cid in node.children_ids:
                    child_sum = _summarize(cid)
                    child_summaries.append(f"Sub-section: {nodes[cid].title}\n{child_sum}")

                additional_text = self._get_block_texts(node.block_ids, document.blocks)

                prompt = _PARENT_SUMMARY_PROMPT.format(
                    title=node.title,
                    sub_summaries="\n\n".join(child_summaries),
                    additional_text=additional_text or "(no additional content)",
                )
            else:
                text = self._get_block_texts(node.block_ids, document.blocks)
                if not text:
                    section_summaries[node_id] = ""
                    return ""

                prompt = _SECTION_SUMMARY_PROMPT.format(title=node.title, text=text)

            summary = self._call_llm(prompt)
            section_summaries[node_id] = summary
            return summary

        # Find root nodes (parent_id is None) and summarize each
        root_ids = [nid for nid, n in nodes.items() if n.parent_id is None]
        for rid in root_ids:
            _summarize(rid)

        # Document summary: combine all root-level section summaries
        root_summaries = []
        for rid in root_ids:
            root_title = nodes[rid].title
            root_sum = section_summaries.get(rid, "")
            if root_sum:
                root_summaries.append(f"Section: {root_title}\n{root_sum}")

        if root_summaries:
            doc_prompt = _DOCUMENT_SUMMARY_PROMPT.format(
                section_summaries="\n\n".join(root_summaries)
            )
            document.summaries.document_summary = self._call_llm(doc_prompt)

        # One-sentence summary
        if document.summaries.document_summary:
            sent_prompt = _ONE_SENTENCE_PROMPT.format(
                document_summary=document.summaries.document_summary
            )
            document.summaries.one_sentence_summary = self._call_llm(sent_prompt)

        document.summaries.section_summaries = section_summaries
        return document

    def _get_block_texts(self, block_ids: list[int], blocks: dict[int, DocumentBlock]) -> str:
        texts = []
        for bid in block_ids:
            block = blocks.get(bid)
            if block and block.text:
                texts.append(block.text)
        return "\n\n".join(texts)

    def _call_llm(self, prompt: str) -> str:
        completion = self._client.chat.completions.create(
            model="ecnu-max",
            messages=[
                {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return completion.choices[0].message.content or ""