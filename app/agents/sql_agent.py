import os 
from dotenv import load_dotenv
from app.core.logger import get_logger

log = get_logger(__name__)
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware.human_in_the_loop import HumanInTheLoopMiddleware
from langchain.agents.middleware.types import AgentState, ContextT
from langgraph.checkpoint.memory import MemorySaver
from langgraph.runtime import Runtime
from typing import Any
import re
from .base_model import get_llm_with_fallbacks
from .rag_helper.config import MODEL_REGISTRY, DEFAULT_TEMPERATURE

load_dotenv()

class ConditionalSqlHumanInTheLoop(HumanInTheLoopMiddleware):
    def _needs_interrupt(self, tool_call: dict) -> bool:
        if tool_call["name"] == "sql_db_query":
            query = tool_call["args"].get("query", "")
            if not query and isinstance(tool_call["args"], str):
                query = tool_call["args"]
            dml_pattern = re.compile(r'\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|REPLACE)\b', re.IGNORECASE)
            needs = bool(dml_pattern.search(str(query)))
            if needs:
                log.warning("🛑 DML query detected — requiring human approval: %s", query[:100])
            return needs
        return True

    def after_model(self, state: AgentState, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        if not messages:
            return None
        last_ai_msg = next((msg for msg in reversed(messages) if getattr(msg, "tool_calls", None)), None)
        if not last_ai_msg or not getattr(last_ai_msg, "tool_calls", None):
            return None

        filtered_interrupt_on = {}
        for tc in last_ai_msg.tool_calls:
            tc_name = tc["name"]
            if tc_name in self.interrupt_on and self._needs_interrupt(tc):
                filtered_interrupt_on[tc_name] = self.interrupt_on[tc_name]

        if not filtered_interrupt_on:
            return None

        temp_middleware = HumanInTheLoopMiddleware(
            interrupt_on=filtered_interrupt_on,
            description_prefix=self.description_prefix
        )
        return temp_middleware.after_model(state, runtime)

    async def aafter_model(self, state: AgentState, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        return self.after_model(state, runtime)

# LangChain models will be loaded lazily to avoid startup delay.


_sql_agent_instance = None

def get_sql_agent():
    global _sql_agent_instance
    if _sql_agent_instance is not None:
        return _sql_agent_instance
    log.info("Initializing SQL agent...")

    _sql_config = MODEL_REGISTRY["sql"]
    toolkit_llm = init_chat_model(
        model=_sql_config["primary"]["model"],
        model_provider=_sql_config["primary"]["provider"],
        temperature=DEFAULT_TEMPERATURE,
    )

    llm = get_llm_with_fallbacks("sql")

    db_url = os.getenv('DATABASE_URL')
    log.info("Connecting SQL agent to database: %s", db_url[:30] + '...' if db_url else 'None')
    db = SQLDatabase.from_uri(db_url)
    toolkit = SQLDatabaseToolkit(db=db, llm=toolkit_llm)
    all_tools = toolkit.get_tools()
    log.info("SQL toolkit loaded %d tools: %s", len(all_tools), [t.name for t in all_tools])

    # laod db schema
    db_schema = db.get_table_info()

    # Only keep the query execution tool — schema is already in the prompt
    tools = [t for t in all_tools if t.name in ("sql_db_query", "sql_db_query_checker")]

    system_prompt = f"""You are a SQL expert. Write and execute SQL directly.
Database schema:
{db_schema}
Use sql_db_query to run your SQL. Always LIMIT results to at most 10 rows unless told otherwise."""

    _sql_agent_instance = create_agent(
        model = llm,
        tools = tools,
        system_prompt=system_prompt,
        middleware=[ 
            ConditionalSqlHumanInTheLoop( 
                interrupt_on={"sql_db_query": True}, 
                description_prefix="SQL modification query requires approval. Please review the query.", 
            ), 
        ], 
        checkpointer=MemorySaver(), 
    )
    log.info("✅ SQL agent created")
    return _sql_agent_instance

@tool
def sql_agent_tool(query: str) -> str:
    """Query the database with natural language and return analytical results."""
    log.info("sql_agent_tool invoked (query_len=%d): %s", len(query), query[:100])
    agent = get_sql_agent()
    result = agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    answer = result["messages"][-1].content
    log.info("✅ sql_agent_tool result (len=%d)", len(answer))
    return answer
