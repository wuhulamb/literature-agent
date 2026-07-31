import json
import re
import unicodedata

from rapidfuzz import fuzz
from pydantic import BaseModel

from openai import OpenAI

from literature_agent.agents.base import BaseAgent
from literature_agent.models.document import (
    Document,
    DocumentBlock,
    DocumentStructure,
    SectionNode,
    BlockHeading,
)
from literature_agent.prompts import load_prompt


# ---------------------------------------------------------------------------
# Pydantic models for structured LLM output
# ---------------------------------------------------------------------------

class IdentifiedHeadingsResult(BaseModel):
    headings: list[str]


class TreeReviewResult(BaseModel):
    is_correct: bool
    missing_headings: list[str] = []
    wrong_headings: list[str] = []


class TitleMatchResult(BaseModel):
    confirmed: bool
    block_id: int | None = None
    level: int | None = None


class BlockHeadingsOutput(BaseModel):
    block_id: int
    headings: list[BlockHeading]


class BatchHeadingsResult(BaseModel):
    blocks: list[BlockHeadingsOutput]


# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------

_IDENTIFY_HEADINGS_PROMPT = load_prompt("structure_identify_headings")
_REVIEW_TREE_PROMPT = load_prompt("structure_review_tree")
_MATCH_TITLE_PROMPT = load_prompt("structure_match_title")
_BATCH_HEADINGS_PROMPT = load_prompt("structure_batch_headings")

_MAX_REVIEW_ITERATIONS = 5


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Normalize text for matching: NFKC, fullwidth→halfwidth, lowercase, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# StructureBuilder (unchanged)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# StructureAgent (rewritten)
# ---------------------------------------------------------------------------

class StructureAgent(BaseAgent):
    """Extract section hierarchy from document blocks.

    Two-phase approach:
      1. Heading identification: full-text LLM → title list → 3-stage matching to blocks.
      2. Review loop: validate tree, fix missing/wrong headings, rebuild, repeat up to 5x.
    """

    def __init__(self, client: OpenAI) -> None:
        self._client = client

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, document: Document) -> Document:
        blocks_list = sorted(document.blocks.values(), key=lambda b: b.id)

        # Clear any existing heading annotations
        for block in blocks_list:
            block.is_heading = False
            block.headings = []

        # -- Step 1: Identify all headings --
        titles = self._identify_all_headings(document)
        print(f"  [structure] LLM identified {len(titles)} heading(s)")

        # Match each title to a block: RapidFuzz → LLM confirm if ambiguous
        title_block_pairs: list[tuple[str, int, int | None]] = []  # (title, block_id, level_or_None)
        used_titles: set[str] = set()

        for title in titles:
            norm = normalize(title)
            if norm in used_titles:
                continue

            normalized_title = normalize(title)

            # Score all blocks with RapidFuzz
            scored: list[tuple[float, DocumentBlock]] = []
            for block in blocks_list:
                norm_source = normalize(block.text or "")
                score = fuzz.partial_ratio(normalized_title, norm_source)
                scored.append((score, block))
            scored.sort(key=lambda x: x[0], reverse=True)

            top_score = scored[0][0] if scored else 0
            top_blocks = [b for s, b in scored if s == top_score]

            if top_score == 100 and len(top_blocks) == 1:
                # Exact match, unambiguous
                block_id = top_blocks[0].id
                document.blocks[block_id].is_heading = True
                title_block_pairs.append((title, block_id, None))
                used_titles.add(norm)
            else:
                # Ambiguous — use LLM to confirm among top 5
                candidates = scored[:5]
                candidate_list = [
                    {"block_id": b.id, "text": b.text[:500]}
                    for _, b in candidates
                ]
                payload = {"title": title, "candidates": candidate_list}
                try:
                    completion = self._client.beta.chat.completions.parse(
                        model="ecnu-max",
                        messages=[
                            {"role": "system", "content": _MATCH_TITLE_PROMPT},
                            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                        ],
                        response_format=TitleMatchResult,
                    )
                except Exception as e:
                    print(f"  [structure] LLM match call failed for '{title}': {e}")
                    continue

                result = completion.choices[0].message.parsed
                if result is not None and result.confirmed and result.block_id is not None:
                    block = document.blocks.get(result.block_id)
                    if block is not None:
                        block.is_heading = True
                        title_block_pairs.append((title, result.block_id, result.level))
                        used_titles.add(norm)
                else:
                    print(f"  [structure] WARNING: could not match heading '{title}', skipping")

        print(f"  [structure] matched {len(title_block_pairs)} heading(s) to blocks")

        # For all is_heading blocks, send them in batch to LLM to generate headings lists
        heading_blocks = [b for b in blocks_list if b.is_heading]
        if heading_blocks:
            batch_payload = [
                {"block_id": b.id, "text": b.text}
                for b in heading_blocks
            ]
            try:
                completion = self._client.beta.chat.completions.parse(
                    model="ecnu-max",
                    messages=[
                        {"role": "system", "content": _BATCH_HEADINGS_PROMPT},
                        {"role": "user", "content": json.dumps(batch_payload, ensure_ascii=False)},
                    ],
                    response_format=BatchHeadingsResult,
                )
            except Exception as e:
                print(f"  [structure] Batch headings LLM call failed: {e}")
                # Fallback: use title_block_pairs directly
                for title, block_id, level in title_block_pairs:
                    block = document.blocks.get(block_id)
                    if block is not None:
                        lvl = level if level is not None else 1
                        block.headings.append(BlockHeading(title=title, level=lvl))
            else:
                result = completion.choices[0].message.parsed
                if result is not None:
                    for b_out in result.blocks:
                        block = document.blocks.get(b_out.block_id)
                        if block is not None:
                            block.headings = b_out.headings
                else:
                    for title, block_id, level in title_block_pairs:
                        block = document.blocks.get(block_id)
                        if block is not None:
                            lvl = level if level is not None else 1
                            block.headings.append(BlockHeading(title=title, level=lvl))
        else:
            print("  [structure] No heading blocks found")

        # -- Step 2: Review loop --
        for iteration in range(_MAX_REVIEW_ITERATIONS):
            builder = self._rebuild_tree(document)
            full_text = "\n\n".join(b.text for b in blocks_list)
            review = self._review_section_tree(builder.nodes, full_text)
            if review.is_correct:
                print(f"  [structure] Tree review passed on iteration {iteration + 1}")
                document.structure = DocumentStructure(nodes=builder.nodes)
                return document

            print(f"  [structure] Review iteration {iteration + 1}: "
                  f"{len(review.missing_headings)} missing, {len(review.wrong_headings)} wrong")

            any_change = False

            # Add missing headings
            for title in review.missing_headings:
                norm = normalize(title)
                if norm in used_titles:
                    continue
                block_id, level = self._match_title_to_block(title, blocks_list)
                if block_id is None:
                    print(f"  [structure] WARNING: could not match missing heading '{title}', skipping")
                    continue
                block = document.blocks[block_id]
                block.is_heading = True
                block.headings.append(BlockHeading(title=title, level=level if level is not None else 1))
                used_titles.add(norm)
                any_change = True

            # Remove wrong headings
            heading_blocks_for_match = [b for b in blocks_list if b.headings]
            for title in review.wrong_headings:
                match_id, _ = self._match_title_to_block(
                    title, heading_blocks_for_match,
                    text_field="heading_title",
                )
                if match_id is None:
                    print(f"  [structure] WARNING: could not locate wrong heading '{title}', skipping")
                    continue
                block = document.blocks.get(match_id)
                if block is None:
                    continue
                block.headings = [h for h in block.headings if normalize(h.title) != normalize(title)]
                if not block.headings:
                    block.is_heading = False
                used_titles.discard(normalize(title))
                any_change = True

            if not any_change:
                print("  [structure] No changes applied, stopping review loop")
                break

        # Final build
        builder = self._rebuild_tree(document)
        document.structure = DocumentStructure(nodes=builder.nodes)
        return document

    # ------------------------------------------------------------------
    # Step 1: Heading identification from full text
    # ------------------------------------------------------------------

    def _identify_all_headings(self, document: Document) -> list[str]:
        """Send full concatenated text to LLM, return list of heading titles."""
        blocks_list = sorted(document.blocks.values(), key=lambda b: b.id)
        full_text = "\n\n".join(b.text for b in blocks_list)

        try:
            completion = self._client.beta.chat.completions.parse(
                model="ecnu-max",
                messages=[
                    {"role": "system", "content": _IDENTIFY_HEADINGS_PROMPT},
                    {"role": "user", "content": full_text},
                ],
                response_format=IdentifiedHeadingsResult,
            )
        except Exception as e:
            print(f"  [structure] LLM call failed for heading identification: {e}")
            return []

        result = completion.choices[0].message.parsed
        if result is None:
            return []
        return result.headings

    # ------------------------------------------------------------------
    # 3-stage title-to-block matching pipeline
    # ------------------------------------------------------------------

    def _match_title_to_block(
        self,
        title: str,
        blocks: list[DocumentBlock],
        text_field: str = "text",
    ) -> tuple[int | None, int | None]:
        """3-stage matching: normalize → RapidFuzz top-5 → LLM confirm.

        Args:
            title: The heading title string to search for.
            blocks: List of DocumentBlock candidates.
            text_field: Which field to match against ('text' or 'heading_title').
                       When 'heading_title', the search text is built from the
                       block's existing heading titles joined by newlines.

        Returns:
            (block_id, level) if confirmed, (None, None) otherwise.
        """
        normalized_title = normalize(title)

        # Stage 1+2: score all blocks with RapidFuzz, take top 5
        scored: list[tuple[float, DocumentBlock]] = []
        for block in blocks:
            if text_field == "heading_title":
                source = "\n".join(h.title for h in block.headings)
            else:
                source = block.text or ""
            norm_source = normalize(source)
            score = fuzz.partial_ratio(normalized_title, norm_source)
            scored.append((score, block))

        scored.sort(key=lambda x: x[0], reverse=True)
        candidates = scored[:5]

        if not candidates:
            return None, None

        # Stage 3: LLM final confirmation
        candidate_list = []
        for _, b in candidates:
            if text_field == "heading_title":
                text = "\n".join(h.title for h in b.headings)
            else:
                text = b.text
            candidate_list.append({"block_id": b.id, "text": text[:500]})

        payload = {
            "title": title,
            "candidates": candidate_list,
        }

        try:
            completion = self._client.beta.chat.completions.parse(
                model="ecnu-max",
                messages=[
                    {"role": "system", "content": _MATCH_TITLE_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                response_format=TitleMatchResult,
            )
        except Exception as e:
            print(f"  [structure] LLM match call failed for '{title}': {e}")
            return None, None

        result = completion.choices[0].message.parsed
        if result is None or not result.confirmed:
            return None, None

        return result.block_id, result.level

    # ------------------------------------------------------------------
    # Step 2: Tree review
    # ------------------------------------------------------------------

    def _review_section_tree(self, nodes: dict[int, SectionNode], full_text: str) -> TreeReviewResult:
        """Send the current section tree + full document text to the LLM for review."""
        tree_data = self._serialize_tree(nodes)
        user_content = json.dumps(tree_data, ensure_ascii=False) + "\n===FULL TEXT===\n" + full_text

        try:
            completion = self._client.beta.chat.completions.parse(
                model="ecnu-max",
                messages=[
                    {"role": "system", "content": _REVIEW_TREE_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format=TreeReviewResult,
            )
        except Exception as e:
            print(f"  [structure] Tree review LLM call failed: {e}")
            return TreeReviewResult(is_correct=True)

        result = completion.choices[0].message.parsed
        if result is None:
            return TreeReviewResult(is_correct=True)
        return result

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def _serialize_tree(self, nodes: dict[int, SectionNode]) -> list[dict]:
        """Serialize section tree as a nested JSON structure for the LLM."""
        root_ids = [nid for nid, n in nodes.items() if n.parent_id is None]
        result = []
        for rid in root_ids:
            result.extend(self._node_to_list(rid, nodes))
        return result

    def _node_to_list(self, node_id: int, nodes: dict[int, SectionNode]) -> list[dict]:
        node = nodes[node_id]
        entry: dict = {
            "id": node.id,
            "title": node.title,
            "level": node.level,
            "block_count": len(node.block_ids),
        }
        if node.children_ids:
            entry["children"] = []
            for cid in node.children_ids:
                entry["children"].extend(self._node_to_list(cid, nodes))
        return [entry]

    # ------------------------------------------------------------------
    # Tree rebuild from annotated blocks
    # ------------------------------------------------------------------

    @staticmethod
    def _rebuild_tree(document: Document) -> StructureBuilder:
        """Rebuild StructureBuilder from Document's block headings list."""
        builder = StructureBuilder()

        if document.metadata.title:
            builder.process_heading(-1, document.metadata.title, 0)

        used: set[str] = set()
        if document.metadata.title:
            used.add(normalize(document.metadata.title))

        for block in sorted(document.blocks.values(), key=lambda b: b.id):
            if block.headings:
                for h in block.headings:
                    norm = normalize(h.title)
                    if norm not in used:
                        builder.process_heading(block.id, h.title, h.level)
                        used.add(norm)
                    else:
                        builder.process_non_heading(block.id)
            else:
                builder.process_non_heading(block.id)

        return builder
