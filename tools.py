import requests
from langchain_core.tools import tool

from tavily import TavilyClient
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API"))


@tool
def Web_search(user=str):
    """Searc the web on recent reliable information on a topic ,Return Titles ,Url """
    result=tavily.search(query= user,max_results=5)


    out=[]
    for r in result[('results')]:
       out.append(
         f"Title: {r['title']}\n"
         f"URL: {r['url']}\n"
         f"Snippet: {r['content'][300:]}\n"
       )
    return "\n---\n".join(out)

fe=Web_search.invoke("what is ai")

from langchain_core.tools import tool
import requests
from bs4 import BeautifulSoup

@tool
def fetch_webpage(url: str):
    """Fetch webpage text."""
    try:
        html = requests.get(url, timeout=10,headers={"user-Agent":"Mozilla/5.0"}).text
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(" ", strip=True)[:3000]
    except Exception as e:
        return f"Error: {e}"
