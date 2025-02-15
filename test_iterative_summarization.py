from src.iterative_summarizer.state import State
from src.iterative_summarizer.graph import graph

test_state = State(url="https://lilianweng.github.io/posts/2023-06-23-agent/")
print(test_state.url)

summary = graph.invoke(test_state)
print(f"Summary result:\n\n{summary}", summary)
print("...")
print("...")
print(summary)