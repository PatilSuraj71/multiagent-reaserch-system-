import os
from dotenv import load_dotenv
load_dotenv()
from tools import Web_search,fetch_webpage

from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI




load_dotenv()

llm = ChatOpenAI(
    model="nvidia/nemotron-3-ultra-550b-a55b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("Nividia_key"),
    temperature=0.2,
)







from langchain_groq import ChatGroq


import os
llm1=ChatGroq(model="llama-3.3-70b-versatile",
             api_key=os.getenv("groq_key"))

def build_search_agent():
    return create_agent(
        model=llm1,
        tools=[Web_search]
    )

def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[fetch_webpage])

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm1 | StrOutputParser()

#critic_chain

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()