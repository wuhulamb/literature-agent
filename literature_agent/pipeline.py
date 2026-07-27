from datetime import datetime

from literature_agent.utils.pdf_reader import parse
from literature_agent.utils.llm_client import get_client
from literature_agent.utils import json_storage
from literature_agent.agents.metadata import MetadataAgent
from literature_agent.agents.structure import StructureAgent
from literature_agent.agents.summary import SummaryAgent


def run(pdf_path: str, output_path: str | None = None) -> None:
    doc = json_storage.load(pdf_path, output_path)

    if doc is None:
        print("[pipeline] Parsing PDF...")
        doc = parse(pdf_path)
        print(f"  -> id: {doc.id}")
        print(f"  -> blocks: {len(doc.blocks)}, pages: {len(doc.content.pages) if doc.content.pages else 0}")

    client = get_client()

    if not doc.processing.meta_done:
        print("[pipeline] Extracting metadata...")
        agent = MetadataAgent(client)
        doc = agent.run(doc)
        doc.processing.meta_done = True
        print(f"  -> title: {doc.metadata.title}")
        json_storage.save(doc, pdf_path, output_path)
    else:
        print("[pipeline] Metadata already done, skipping.")

    if not doc.processing.structure_done:
        print("[pipeline] Extracting structure...")
        agent = StructureAgent(client)
        doc = agent.run(doc)
        doc.processing.structure_done = True
        node_count = len(doc.structure.nodes) if doc.structure else 0
        print(f"  -> section nodes: {node_count}")
        json_storage.save(doc, pdf_path, output_path)
    else:
        print("[pipeline] Structure already done, skipping.")

    if not doc.processing.summary_done:
        print("[pipeline] Generating summaries...")
        agent = SummaryAgent(client)
        doc = agent.run(doc)
        doc.processing.summary_done = True
        print(f"  -> document_summary: {'done' if doc.summaries.document_summary else 'none'}")
    else:
        print("[pipeline] Summaries already done, skipping.")

    doc.processing.last_updated = datetime.now()
    out_path = json_storage.save(doc, pdf_path, output_path)
    print(f"[pipeline] Done. Document saved to {out_path}")