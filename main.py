import argparse
from pathlib import Path

# Ensure literature_agent package is discoverable
import sys

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from literature_agent.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Process a PDF document through the literature agent pipeline.")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("-o", "--output", help="Output JSON path (default: {pdf_dir}/{pdf_name}.document.json)")
    args = parser.parse_args()

    run(args.pdf_path, args.output)


if __name__ == "__main__":
    main()