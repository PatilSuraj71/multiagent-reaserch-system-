import os
from dotenv import load_dotenv
load_dotenv()
from tools import Web_search,fetch_webpage
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
llm=ChatGroq(model="llama-3.3-70b-versatile",
             api_key=os.getenv("groq_key"))

llm1=ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("KEY"))



def build_search_agent():
    return create_agent(
        model=llm,
        tools=[Web_search]
    )

def build_reader_agent():
    return create_agent(
        model=llm1,
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