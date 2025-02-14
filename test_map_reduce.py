from src.map_reduce_summarizer.state import OverallState, SummaryState
from src.map_reduce_summarizer.graph import map_reduce_graph


summary_state = SummaryState(content="test")
print(summary_state.content)

starting_state = OverallState(url="https://langchain-ai.github.io/langgraph/troubleshooting/errors/")
print(starting_state.url)
summary = map_reduce_graph.invoke(starting_state)