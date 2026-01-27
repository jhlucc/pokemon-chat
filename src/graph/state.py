import operator
from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    The state of the agent graph.
    """

    # The history of messages in the conversation
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # The next node to route to
    next: str

    # Optional: specialized state keys
    # rag_query: str
    # documents: List[Document]
