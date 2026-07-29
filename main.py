import argparse
from pathlib import Path

# Ensure literature_agent package is discoverable
import sys

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from literature_agent.pipeline import run
from literature_agent.storage import save as save_json, load as load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Process a PDF document through the literature agent pipeline.")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("-o", "--output", help="Output JSON path (default: {pdf_dir}/{pdf_name}.document.json)")
    args = parser.parse_args()

    pdf_bytes = Path(args.pdf_path).read_bytes()

    # Try to resume from existing checkpoint
    doc = load_json(args.pdf_path, args.output)

    doc = run(pdf_bytes, document=doc, checkpoint=lambda d: save_json(d, args.pdf_path, args.output))

    out_path = save_json(doc, args.pdf_path, args.output)
    print(f"[pipeline] Done. Document saved to {out_path}")


if __name__ == "__main__":
    main()