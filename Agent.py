import os
from dotenv import load_dotenv

load_dotenv()

from tools import Web_search, fetch_webpage

from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq


# =========================================================
# MODELS
# =========================================================

# OpenRouter / NVIDIA
llm = ChatOpenAI(
    model="nvidia/nemotron-3-ultra-550b-a55b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("Nividia_key"),
    temperature=0.2,
)

# Groq
llm1 = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("groq_key"),
    temperature=0.2,
)


# =========================================================
# SEARCH AGENT
# =========================================================

def build_search_agent():

    return create_agent(
        model=llm1,
        tools=[Web_search],
        system_prompt="""
You are a professional web research agent.

Your job is to research the user's topic thoroughly.

Rules:
1. Search for multiple relevant sources.
2. Prefer reliable and authoritative sources.
3. Do not rely on a single webpage.
4. Extract specific facts, numbers, dates and evidence.
5. Return useful research findings, not generic explanations.
6. Include the URLs of the sources you found.
7. If sources disagree, explicitly mention the disagreement.
"""
    )


# =========================================================
# READER AGENT
# =========================================================

def build_reader_agent():

    return create_agent(
        model=llm,
        tools=[fetch_webpage],
        system_prompt="""
You are a professional research analyst.

Your job is to read webpages provided by the research process.

For every useful source:
- Identify the main claims.
- Extract important facts.
- Extract statistics and dates.
- Identify evidence supporting the claims.
- Ignore advertisements and irrelevant information.
- Do not invent information.
- Keep the source URL.
- Clearly distinguish facts from opinions.

Return structured research notes that another writer can use.
"""
    )


# =========================================================
# WRITER
# =========================================================

writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a senior professional research writer.

Your job is to transform raw research into a high-quality,
fact-based research report.

IMPORTANT RULES:

1. Use ONLY information contained in the research.
2. Do not invent facts, statistics, companies, dates or URLs.
3. Synthesize information from multiple sources.
4. Do not simply copy or repeat the research.
5. Explain WHY each important finding matters.
6. Compare conflicting information when sources disagree.
7. Use specific evidence whenever available.
8. Avoid generic filler.
9. Make every section informative.
10. Write like a professional research analyst.

The final report must be detailed but easy to read.
"""
    ),
    (
        "human",
        """
Research Topic:
{topic}

Research Gathered:
{research}

Write a comprehensive research report using the following structure:

# Introduction

Explain the topic and why it matters.

# Key Findings

Provide at least 5 important findings.

For every finding:

- State the finding clearly.
- Explain it.
- Provide supporting evidence.
- Explain why it matters.

# Analysis

Compare the major findings and explain the broader implications.

# Limitations

Mention missing information, source limitations,
conflicting evidence or uncertainty.

# Conclusion

Give a concise evidence-based conclusion.

# Sources

List the URLs of the sources used.

Do not make up sources or URLs.
"""
    ),
])

writer_chain = writer_prompt | llm1 | StrOutputParser()


# =========================================================
# CRITIC
# =========================================================

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a senior research editor.

Critically evaluate the report for:

- factual accuracy
- depth
- evidence
- logical reasoning
- source quality
- completeness
- structure
- clarity
- unsupported claims
- hallucinations
- repetition

Be strict. Do not give a high score simply because
the writing sounds professional.
"""
    ),
    (
        "human",
        """
Evaluate this research report:

{report}

Return:

Score: X/10

Factual Accuracy:
- ...

Research Depth:
- ...

Evidence Quality:
- ...

Structure:
- ...

Strengths:
- ...
- ...

Problems:
- ...
- ...

Missing Information:
- ...
- ...

Recommended Improvements:
- ...
- ...

Final Verdict:
...
"""
    ),
])

critic_chain = critic_prompt | llm | StrOutputParser()