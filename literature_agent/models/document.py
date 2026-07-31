from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class Language(str, Enum):
    EN = "en"
    ZH = "zh"
    JA = "ja"


class Page(BaseModel):
    page_number: int
    text: str


class BlockHeading(BaseModel):
    title: str
    level: int


class DocumentBlock(BaseModel):
    id: int
    text: str
    page: int | None = None
    is_heading: bool = False
    headings: list[BlockHeading] = []


class SectionNode(BaseModel):
    id: int
    title: str
    level: int
    parent_id: int | None = None
    children_ids: list[int] = []
    block_ids: list[int] = []


class ProcessingState(BaseModel):
    meta_done: bool = False
    structure_done: bool = False
    summary_done: bool = False
    last_updated: datetime | None = None


class DocumentMetadata(BaseModel):
    title: str | None = None
    authors: list[str] = []
    year: int | None = None
    journal: str | None = None
    language: Language | None = None


class DocumentContent(BaseModel):
    full_text: str = ""
    pages: list[Page] | None = None


class DocumentStructure(BaseModel):
    nodes: dict[int, SectionNode] = {}


class DocumentSummary(BaseModel):
    section_summaries: dict[int, str] = {}  # key 为 section id
    document_summary: str | None = None
    one_sentence_summary: str | None = None


class Document(BaseModel):
    id: str
    metadata: DocumentMetadata = DocumentMetadata()
    content: DocumentContent = DocumentContent()
    blocks: dict[int, DocumentBlock] = {}
    structure: DocumentStructure | None = None
    summaries: DocumentSummary = DocumentSummary()
    processing: ProcessingState = ProcessingState()