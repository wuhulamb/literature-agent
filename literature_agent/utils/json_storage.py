import json
from pathlib import Path

from literature_agent.models.document import Document


def _default_output_path(pdf_path: str) -> Path:
    pdf = Path(pdf_path)
    return pdf.with_name(f"{pdf.stem}.document.json")


def save(document: Document, pdf_path: str | None = None, output_path: str | None = None) -> Path:
    """Save Document to JSON.

    Args:
        document: The Document to save.
        pdf_path: Used to derive default output path if output_path is not given.
        output_path: Explicit output path. If not provided, derived from pdf_path.
    """
    if output_path is not None:
        out = Path(output_path)
    else:
        path = pdf_path or document.path
        out = _default_output_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def load(pdf_path: str, output_path: str | None = None) -> Document | None:
    """Load Document from JSON file.

    Args:
        pdf_path: Used to derive default JSON path if output_path is not given.
        output_path: Explicit JSON path to load from.
    """
    if output_path is not None:
        path = Path(output_path)
    else:
        path = _default_output_path(pdf_path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return Document.model_validate(data)