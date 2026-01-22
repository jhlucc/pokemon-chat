import json
import time
from typing import Any, Dict, List, Optional
from uuid import UUID
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from src.core.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

class FileTraceCallbackHandler(BaseCallbackHandler):
    """
    Logs LangChain execution traces to a JSONL file for local debugging.
    Logs inputs, outputs, and latency.
    """
    def __init__(self, filename: str = "trace.jsonl"):
        self.log_path = settings.paths.log_dir / filename
        self.starts: Dict[str, float] = {}
        
    def _log(self, data: Dict[str, Any]):
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to log trace: {e}")

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> Any:
        self.starts[str(run_id)] = time.time()
        self._log({
            "type": "chain_start",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "name": serialized.get("name", "Chain"),
            "inputs": inputs,
            "timestamp": time.time()
        })

    def on_chain_end(
        self, outputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> Any:
        duration = time.time() - self.starts.get(str(run_id), time.time())
        self._log({
            "type": "chain_end",
            "run_id": str(run_id),
            "outputs": outputs,
            "duration": duration,
            "timestamp": time.time()
        })

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> Any:
        self.starts[str(run_id)] = time.time()
        self._log({
            "type": "llm_start",
            "run_id": str(run_id),
            "prompts": prompts,
            "timestamp": time.time()
        })

    def on_llm_end(
        self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> Any:
        duration = time.time() - self.starts.get(str(run_id), time.time())
        self._log({
            "type": "llm_end",
            "run_id": str(run_id),
            "generated": [g[0].text for g in response.generations],
            "duration": duration,
            "timestamp": time.time()
        })
