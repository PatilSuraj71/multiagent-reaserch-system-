import os

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from tools import Web_search, fetch_webpage


load_dotenv()


# ============================================================
# OPENROUTER MODEL
# ============================================================

llm = ChatOpenAI(
    model="nvidia/nemotron-3-ultra-550b-a55b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("Nividia_key"),
)


# ============================================================
# GROQ MODEL
# ============================================================

llm1 = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("groq_key"),
)


# ============================================================
# SEARCH AGENT
# ============================================================

def build_search_agent():

    return create_agent(
        model=llm1,
        tools=[Web_search],
    )


# ============================================================
# READER AGENT
# ============================================================

def build_reader_agent():

    return create_agent(
        model=llm,
        tools=[fetch_webpage],
    )


# ============================================================
# WRITER CHAIN
# ============================================================

writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an expert research writer.

        Write clear, structured, factual and insightful
        research reports.
        """
    ),

    (
        "human",
        """
        Write a detailed research report on the topic below.

        Topic:
        {topic}

        Research Gathered:
        {research}

        Structure the report as:

        1. Introduction
        2. Key Findings
           - Minimum 3 well-explained points
        3. Conclusion
        4. Sources
           - List all URLs found in the research

        Be detailed, factual and professional.
        """
    ),
])


writer_chain = (
    writer_prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# CRITIC CHAIN
# ============================================================

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a sharp and constructive research critic.

        Be honest, specific and analytical.
        """
    ),

    (
        "human",
        """
        Review the research report below and evaluate it strictly.

        Report:
        {report}

        Respond in exactly this format:

        Score: X/10

        Strengths:
        - ...
        - ...

        Areas to Improve:
        - ...
        - ...

        One line verdict:
        ...
        """
    ),
])


critic_chain = (
    critic_prompt
    | llm1
    | StrOutputParser()
)