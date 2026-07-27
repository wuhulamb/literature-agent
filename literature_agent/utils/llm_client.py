import os

from dotenv import load_dotenv
from openai import OpenAI


def get_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("CHATECNU_API_KEY")
    if not api_key:
        raise ValueError("CHATECNU_API_KEY not found in .env file")
    return OpenAI(
        api_key=api_key,
        base_url="https://chat.ecnu.edu.cn/open/api/v1",
    )