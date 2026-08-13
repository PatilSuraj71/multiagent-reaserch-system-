import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from tavily import TavilyClient
from langchain_core.tools import tool

load_dotenv()

tavily = TavilyClient(
    api_key=os.getenv("TAVILY_API")
)


@tool
def Web_search(user: str):
    """Search the web for recent and reliable information.
    Return titles, URLs, and snippets.
    """

    try:
        result = tavily.search(
            query=user,
            max_results=5
        )

        out = []

        for r in result["results"]:
            out.append(
                f"Title: {r.get('title', '')}\n"
                f"URL: {r.get('url', '')}\n"
                f"Snippet: {r.get('content', '')[:1000]}\n"
            )

        return "\n---\n".join(out)

    except Exception as e:
        return f"Search error: {str(e)}"


@tool
def fetch_webpage(url: str):
    """Fetch webpage text from a URL."""

    try:
        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        text = soup.get_text(
            " ",
            strip=True
        )

        return text[:5000]

    except Exception as e:
        return f"Error fetching webpage: {str(e)}"