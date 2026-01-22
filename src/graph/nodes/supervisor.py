from typing import List, Literal, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.messages import SystemMessage
from pydantic import BaseModel

from src.core.settings import settings
from src.graph.state import AgentState

# Define the set of workers
# Definition of workers
AVAILABLE_WORKERS = ["rag_worker", "web_worker", "graph_worker", "stats_worker", "mcp_worker"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH.\n\n"
    "Worker capabilities:\n"
    "- rag_worker: Pokemon knowledge retrieval from vector database\n"
    "- web_worker: Real-time web search for current information\n"
    "- graph_worker: Knowledge graph queries (relationships, evolutions)\n"
    "- stats_worker: Pokemon stats and battle calculations\n"
    "- mcp_worker: Geographic location queries (real-world Pokemon locations)"
)

options = ["FINISH"] + AVAILABLE_WORKERS

function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(AVAILABLE_WORKERS))


class RouteResponse(BaseModel):
    next: Literal["FINISH", "rag_worker", "web_worker", "graph_worker", "stats_worker", "mcp_worker"]

class SupervisorNode:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.llm.model_name,
            api_key=settings.llm.api_key,
            base_url=settings.llm.api_base
        )
        
    def __call__(self, state: AgentState):
        chain = (
            prompt
            | self.llm.with_structured_output(RouteResponse)
        )
        result = chain.invoke(state)
        # Convert Pydantic model to dict for state update
        return {"next": result.next}

def supervisor_node(state: AgentState):
    node = SupervisorNode()
    return node(state)
