import operator
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from typing import List, Literal, TypedDict, Annotated

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    collapse_docs,
    split_list_of_docs,
)

from map_reduce_summarizer.state import SummaryState, OverallState
from text_utils import get_docs_from_url, split_text

llm = ChatOllama(model="llama3.2", temperature=0)
token_max = 1000

# Prompts
map_prompt = ChatPromptTemplate.from_messages(
    [("system", "Write a concise summary of the following:\\n\\n{context}")]
)

reduce_template = """
The following is a set of summaries:
{docs}
Take these and distill it into a final, consolidated summary
of the main themes.
"""
# reduce_prompt = ChatPromptTemplate.from_messages(
#     [("human",reduce_template)]
# )
reduce_prompt = ChatPromptTemplate([("human", reduce_template)])


# Util functions
def length_function(documents: List[Document]) -> int:
    """Get the number of tokens for input contents"""
    return sum(llm.get_num_tokens(doc.page_content) for doc in documents)

def _reduce(input: dict) -> str:
    prompt = reduce_prompt.invoke(input)
    response = llm.invoke(prompt)
    return response.content

# Graph functions
def get_content(state: OverallState):
    docs = get_docs_from_url(state.url)  #(state["url"])
    split_docs = split_text(docs)
    return {
        "contents": [doc.page_content for doc in split_docs]
    }

def map_summaries(state: OverallState):
    """Map summaries to the contents
    Args:
        state: OverallState

    Returns:
        List of Send objects, each consist of the name of a node in the graph and the state to be sent to that node
    """

    docs = get_docs_from_url(state.url)  #(state["url"])
    split_docs = split_text(docs)
    
    retrieved_content = [doc.page_content for doc in split_docs]
    state.contents = retrieved_content

    return [
         Send("generate_summary", {"content": content}) for content in retrieved_content 
        #Send("generate_summary", SummaryState(content=content)) for content in state.contents # {"content": content}) for content in state.contents # state["contents"]
    ]
    #return [SummaryState(content=content) for content in state["contents"]]
    # for content in state["contents"]:
    #     yield SummaryState(content=content)

def generate_summary(state: SummaryState):
    """Generate a summary for a given content"""

    prompt = map_prompt.invoke(state["content"]) #state.content) #(state["content"])
    response = llm.invoke(prompt)
    return {"summaries": [response.content]}


def collect_summaries(state: OverallState):
    return {
        "collapsed_summaries": [Document(summary) for summary in state.summaries] # state["summaries"]]
    }

def should_collapse(state: OverallState) -> Literal["collapse_summaries", "generate_final_summary"]: 
    """Determine if the summaries should be collapsed or if the final summary should be generated"""
    num_tokens = length_function(state.collapsed_summaries) #(state["collapsed_summaries"])
    if num_tokens > token_max:
        return "collapse_summaries"
    else:
        return "generate_final_summary"

def collapse_summaries(state: OverallState):
    """Collapse the summaries into a final summary"""
    doc_lists = split_list_of_docs(state.collapsed_summaries, length_function, token_max) #(state["collapsed_summaries"], length_function, token_max)
    results = []
    for doc_list in doc_lists:
        results.append(collapse_docs(doc_list, _reduce))

    return {"collapsed_summaries": results}
    
def generate_final_summary(state: OverallState):
    """Generate the final summary"""
    response = _reduce(state.collapsed_summaries) #(state["collapsed_summaries"])
    return {"final_summary": response}

# Graph definition

builder = StateGraph(OverallState)
# nodes
builder.add_node("get_content", get_content)
#builder.add_node("map_summaries", map_summaries)
builder.add_node("generate_summary", generate_summary)
builder.add_node("collect_summaries", collect_summaries)
builder.add_node("collapse_summaries", collapse_summaries)
builder.add_node("generate_final_summary", generate_final_summary)

# edges
#builder.add_edge(START, "get_content")
#builder.add_edge("get_content", "map_summaries")
#builder.add_conditional_edges("map_summaries", [generate_summary])

builder.add_conditional_edges(START, map_summaries, ["generate_summary"])
builder.add_edge("generate_summary", "collect_summaries")
builder.add_conditional_edges("collect_summaries", should_collapse)
builder.add_conditional_edges("collapse_summaries", should_collapse)
builder.add_edge("generate_final_summary", END)

map_reduce_graph = builder.compile()
