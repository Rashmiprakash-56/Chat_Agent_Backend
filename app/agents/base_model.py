from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from .rag_helper.config import MODEL_REGISTRY, DEFAULT_TEMPERATURE
from langchain_core.callbacks import BaseCallbackHandler
from typing import Any, Dict, List

from app.core.logger import get_logger

load_dotenv()
log = get_logger(__name__)

class ModelTrackingCallbackHandler(BaseCallbackHandler):
    """Callback Handler that prints the model name when an LLM starts."""
    
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        model_name = kwargs.get("invocation_params", {}).get("model_name")
        # invoked_params might be different depending on the provider (e.g. valid for ChatOpenAI/ChatGroq)
        if not model_name: 
             model_name = kwargs.get("invocation_params", {}).get("model")
        
        # If still not found, try serialized
        if not model_name and "name" in serialized:
             model_name = serialized["name"]
             
        if model_name:
            log.info(" [Model Tracker] Using model: %s", model_name)
        else:
            log.warning(" [Model Tracker] Using model: (unknown — check serialized: %s)", serialized)

def get_llm_with_fallbacks(agent_name: str):
    """
    Build an LLM with automatic fallbacks for the given agent.
    """
    log.info("Building LLM chain for '%s' (primary: %s, fallbacks: %d)",
             agent_name,
             MODEL_REGISTRY[agent_name]["primary"]["model"],
             len(MODEL_REGISTRY[agent_name]["fallbacks"]))
    config = MODEL_REGISTRY[agent_name]
    tracker = ModelTrackingCallbackHandler()

    # Primary LLM
    primary_llm = init_chat_model(
        model=config["primary"]["model"],
        model_provider=config["primary"]["provider"],
        temperature=DEFAULT_TEMPERATURE,
        callbacks=[tracker]
    )

    # Fallback LLMs
    fallback_llms = []
    for fb in config["fallbacks"]:
        llm = init_chat_model(
            model=fb["model"],
            model_provider=fb["provider"],
            temperature=DEFAULT_TEMPERATURE,
            callbacks=[tracker]
        )
        fallback_llms.append(llm)

    if fallback_llms:
        return primary_llm.with_fallbacks(fallback_llms)
    return primary_llm
