from pydantic import BaseModel, Field
from typing import List, Literal, Union, Optional,Any

class SearchResponse(BaseModel):
    answer: str = Field(description="The final answer to the user's question, synthesizing information from search results.")
    source_urls: List[str] = Field(description="A list of source URLs used to generate the answer.")

class CodeResponse(BaseModel):
    language: str = Field(description="The programming language of the code snippet (e.g., 'python', 'sql').")
    code: str = Field(description="The code snippet itself.")
    explanation: str = Field(description="A brief explanation of what the code does and how to use it.")


class AgentEvent(BaseModel):
    """
    One event in the agent stream. 

    type        | meaning
    ------------|-------------------------------------------------------------
    "log"       | Intermediate progress update (tool usage, thoughts).
                |   content → str
    "answer"    | Final structured answer. Stream ends after this.
                |   response_type → "text" | "code"
                |   content      → SearchResponse | CodeResponse (as dict)
    "error"     | Unrecoverable failure.
                |   content → str (human-readable)
                |   detail  → str | None (full traceback, omit in production)
    """

    type: Literal["log", "answer", "error"]
    content: Any = None           # str for log/error; dict for answer
    response_type: str | None = None   # only set when type == "answer"
    detail: str | None = None          # traceback, only set when type == "error"
