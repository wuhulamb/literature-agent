from datetime import datetime
from collections.abc import Callable

from literature_agent.utils.pdf_reader import parse
from literature_agent.utils.llm_client import get_client
from literature_agent.agents.metadata import MetadataAgent
from literature_agent.agents.structure import StructureAgent
from literature_agent.agents.summary import SummaryAgent
from literature_agent.models.document import Document


def run(
    pdf_bytes: bytes,
    document: Document | None = None,
    *,
    checkpoint: Callable[[Document], None] = lambda _: None,
) -> Document:
    """Process PDF bytes through the pipeline.

    Args:
        pdf_bytes: Raw PDF content.
        document: Existing document to resume from. If None, a new document is
                  created by parsing the PDF. Completed steps (based on
                  ProcessingState) are skipped.
        checkpoint: Hook called after each completed step with the current
                    document. No-op by default.

    Returns:
        Fully processed Document with ProcessingState marking completion.
    """
    if document is None:
        print("[pipeline] Parsing PDF...")
        doc = parse(pdf_bytes)
        print(f"  -> id: {doc.id}")
        print(f"  -> blocks: {len(doc.blocks)}, pages: {len(doc.content.pages) if doc.content.pages else 0}")
    else:
        doc = document
        print(f"[pipeline] Resuming document {doc.id}...")

    client = get_client()

    if not doc.processing.meta_done:
        print("[pipeline] Extracting metadata...")
        agent = MetadataAgent(client)
        doc = agent.run(doc)
        doc.processing.meta_done = True
        print(f"  -> title: {doc.metadata.title}")
        checkpoint(doc)
    else:
        print("[pipeline] Metadata already done, skipping.")

    if not doc.processing.structure_done:
        print("[pipeline] Extracting structure...")
        agent = StructureAgent(client)
        doc = agent.run(doc)
        doc.processing.structure_done = True
        node_count = len(doc.structure.nodes) if doc.structure else 0
        print(f"  -> section nodes: {node_count}")
        checkpoint(doc)
    else:
        print("[pipeline] Structure already done, skipping.")

    if not doc.processing.summary_done:
        print("[pipeline] Generating summaries...")
        agent = SummaryAgent(client)
        doc = agent.run(doc)
        doc.processing.summary_done = True
        print(f"  -> document_summary: {'done' if doc.summaries.document_summary else 'none'}")
        checkpoint(doc)
    else:
        print("[pipeline] Summaries already done, skipping.")

    doc.processing.last_updated = datetime.now()
    return doc