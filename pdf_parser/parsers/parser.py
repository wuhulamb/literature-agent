import pymupdf
from pdf_parser.models import Document, Page, TextBlock, ImageBlock, Block


def _to_bbox(raw: list[float]) -> tuple[float, float, float, float]:
    return (raw[0], raw[1], raw[2], raw[3])


def _extract_text_block(page_id: int, block_id: int, global_id_start: int, raw: dict) -> tuple[TextBlock, int]:
    """从 PyMuPDF 文本 block 原始数据提取 TextBlock，返回 (block, 下一个全局 id)"""
    lines = raw["lines"]
    text_parts = []
    for line in lines:
        for span in line["spans"]:
            text_parts.append(span["text"])
    full_text = " ".join(text_parts)

    tb = TextBlock(
        id=global_id_start,
        bbox=_to_bbox(raw["bbox"]),
        text=full_text,
    )
    return tb, global_id_start + 1


def _extract_image_block(page_id: int, block_id: int, global_id_start: int, raw: dict) -> tuple[ImageBlock, int]:
    """从 PyMuPDF 图片 block 原始数据提取 ImageBlock"""
    ib = ImageBlock(
        id=global_id_start,
        bbox=_to_bbox(raw["bbox"]),
        width=raw["width"],
        height=raw["height"],
        ext=raw["ext"],
    )
    return ib, global_id_start + 1


BLOCK_TYPE_MAP = {
    0: _extract_text_block,
    1: _extract_image_block,
}


def _extract_page(page_id: int, raw_page, global_id_start: int) -> tuple[Page, int]:
    """提取单个页面，返回 (Page, 下一个全局 id)"""
    raw_blocks = raw_page.get_text("dict")["blocks"]
    blocks: list[Block] = []
    next_id = global_id_start

    for block_id, raw in enumerate(raw_blocks):
        raw_type = raw.get("type")
        extractor = BLOCK_TYPE_MAP.get(raw_type)
        if extractor is None:
            continue
        block, next_id = extractor(page_id, block_id, next_id, raw)
        blocks.append(block)

    return Page(id=page_id, blocks=blocks), next_id


def parse_pdf(path: str | bytes) -> Document:
    """解析 PDF 文件，返回 Document 模型"""
    if isinstance(path, bytes):
        doc = pymupdf.open(stream=path, filetype="pdf")
        source = ""
    else:
        doc = pymupdf.open(path)
        source = path
    pages: list[Page] = []
    global_id = 0

    for page_id in range(len(doc)):
        raw_page = doc[page_id]
        page, global_id = _extract_page(page_id, raw_page, global_id)
        pages.append(page)

    result = Document(
        path=source,
        pages=pages,
        total_pages=len(doc),
    )

    doc.close()
    return result