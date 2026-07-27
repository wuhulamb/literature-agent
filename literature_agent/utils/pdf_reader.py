from pathlib import Path
from uuid import uuid4

from pdf_parser.parsers import parse_pdf
from pdf_parser.models import TextBlock as RawTextBlock

from literature_agent.models.document import (
    Document,
    DocumentBlock,
    DocumentContent,
    Page,
)


def parse(pdf_path: str) -> Document:
    """Parse a PDF file and return a Document with content and blocks populated."""
    raw_doc = parse_pdf(pdf_path)
    pdf_path_obj = Path(pdf_path)

    full_text_parts: list[str] = []
    pages: list[Page] = []
    blocks: dict[int, DocumentBlock] = {}

    for raw_page in raw_doc.pages:
        page_text_parts: list[str] = []
        for raw_block in raw_page.blocks:
            if isinstance(raw_block, RawTextBlock):
                block = DocumentBlock(
                    id=raw_block.id,
                    text=raw_block.text,
                    page=raw_page.id,
                )
                blocks[block.id] = block
                page_text_parts.append(raw_block.text)

        page_text = "\n\n".join(page_text_parts)
        pages.append(Page(page_number=raw_page.id, text=page_text))
        full_text_parts.append(page_text)

    return Document(
        id=str(uuid4()),
        path=str(pdf_path_obj.resolve()),
        content=DocumentContent(
            full_text="\n\n".join(full_text_parts),
            pages=pages,
        ),
        blocks=blocks,
    )