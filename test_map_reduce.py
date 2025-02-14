from src.map_reduce_summarizer.state import OverallState, SummaryState
from src.map_reduce_summarizer.graph import map_reduce_graph
from src.text_utils import split_text, get_docs_from_url


summary_state = SummaryState(content="test")
print(summary_state.content)

starting_state = OverallState(url="https://langchain-ai.github.io/langgraph/troubleshooting/errors/")
print(starting_state.url)
#

docs = get_docs_from_url(starting_state.url)  #(state["url"])
split_docs = split_text(docs)

retrieved_content = [doc.page_content for doc in split_docs]

print(len(retrieved_content))
summary = map_reduce_graph.invoke(starting_state)

print("done")