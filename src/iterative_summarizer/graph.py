import os
from typing import Literal

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from src.iterative_summarizer.state import State
from src.text_utils import get_split_page_content_from_url


llm = ChatOllama(model="phi4", temperature=0)


# prompts
summarize_prompt = ChatPromptTemplate(
    [("human", "Write a concise summary of the following: {context}")]
)
initial_summary_chain = summarize_prompt | llm | StrOutputParser()

refine_template = """
Produce a final summary.

Existing summary up to this point:
{existing_answer}

New context:
-------------
{context}
-------------

Given the new context, refine the original summary
"""

refine_prompt = ChatPromptTemplate([("human", refine_template)])
refine_summary_chain= refine_prompt | llm | StrOutputParser()


# Graph Nodes
def generate_initial_summary(state: State, config: RunnableConfig):
    if state.contents is None:
       state.contents = get_split_page_content_from_url(state.url)

    summary = initial_summary_chain.invoke(
        state.contents[0],
        config
    )
    return {
        "summary": summary,
        "index": 1,
        "contents": state.contents
    }

def refine_summary(state: State, config: RunnableConfig):
    content = state.contents[state.index]
    summary = refine_summary_chain.invoke(
        {"existing_answer": state.summary, "context": content},
        config
    )

    return {"summary": summary, "index": state.index + 1}

def should_refine (state: State) -> Literal["refine_summary", END]:
    if state.index >= len(state.contents):
        return END
    else:
        return "refine_summary"

# Define nodes
builder = StateGraph(State)
builder.add_node("generate_initial_summary", generate_initial_summary)
builder.add_node("refine_summary", refine_summary)

# Define Edges
builder.add_edge(START, "generate_initial_summary")
builder.add_conditional_edges("generate_initial_summary", should_refine)
builder.add_conditional_edges("refine_summary", should_refine)

graph = builder.compile()