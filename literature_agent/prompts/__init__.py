from pathlib import Path


_PROMPT_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """Load a prompt template from prompts/{name}.txt."""
    path = _PROMPT_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8")


__all__ = ["load_prompt"]