from dataclasses import dataclass, field
from typing import List


@dataclass(kw_only=True)
class State:
    contents: List[str] = field(default=None)
    index: int = field(default=0)
    summary: str = field(default=None)
    url: str = field(default=None)