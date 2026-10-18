import operator
from typing import TypedDict

from langgraph.graph.message import Annotated


class AgentState(TypedDict):
    file_path: str
    embedding: list[float] | None
    duration: float | None
    current_metadata: dict[str, str]
    db_match: dict[str, str] | None
    proposed_metadata: dict[str, str]
    review_status: str
    messages: Annotated[list[str], operator.add]
