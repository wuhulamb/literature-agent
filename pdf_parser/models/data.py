from pydantic import BaseModel
from typing import Literal


class TextBlock(BaseModel):
    """文本 block，包含一个段落/标题等连续文本"""
    id: int
    type: Literal["text"] = "text"
    bbox: tuple[float, float, float, float]
    text: str


class ImageBlock(BaseModel):
    """图片 block"""
    id: int
    type: Literal["image"] = "image"
    bbox: tuple[float, float, float, float]
    width: int
    height: int
    ext: str


Block = TextBlock | ImageBlock


class Page(BaseModel):
    """PDF 页面，包含该页所有 block"""
    id: int
    blocks: list[Block]


class Document(BaseModel):
    """PDF 文档，包含所有页面"""
    path: str = ""
    pages: list[Page]
    total_pages: int
