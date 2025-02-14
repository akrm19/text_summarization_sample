import operator

from dataclasses import dataclass, field
from typing import List, Annotated, Literal, TypedDict

from langchain_core.documents import Document

@dataclass(kw_only=True)
class OverallState():
    url: str = field(default=None)
    contents: List[str] = field(default=None)
    summaries: Annotated[list, operator.add] = field(default_factory=list)
    collapsed_summaries: List[Document] = field(default=None)
    final_summary: str = field(default=None)

@dataclass(kw_only=True)
class SummaryState():
    content: str = field(default=None)
