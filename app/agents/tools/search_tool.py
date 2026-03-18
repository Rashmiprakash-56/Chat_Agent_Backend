import os
from typing import List, Optional
from langchain_core.tools import tool
from langchain_tavily import (
    TavilyCrawl,
    TavilyExtract,
    TavilyGetResearch,
    TavilyMap,
    TavilyResearch,
    TavilySearch,
)
from dotenv import load_dotenv
from app.core.logger import get_logger

load_dotenv()
log = get_logger(__name__)

"""
LangChain @tool wrappers for every Tavily API endpoint.
Endpoints covered
-----------------
1. /search      → search_tool         (TavilySearch)
2. /extract     → extract_tool        (TavilyExtract)
3. /crawl       → crawl_tool          (TavilyCrawl)
4. /map         → map_tool            (TavilyMap)
5. /research    → research_tool       (TavilyResearch)   creates async task
6. /research    → get_research_tool   (TavilyGetResearch) polls result by ID

"""

# ---------------------------------------------------------------------------
# Make sure TAVILY_API_KEY is set before importing
# ---------------------------------------------------------------------------
if not os.getenv("TAVILY_API_KEY"):
    raise EnvironmentError(
        "TAVILY_API_KEY environment variable is not set. "
    )


# ===========================================================================
# 1. SEARCH  –  /search
#    Real-time web search optimised for LLMs.
# ===========================================================================
_search_web = TavilySearch(
    max_results=3,              
    search_depth="advanced",        
    include_answer=True,         
    include_raw_content=False,   
)


@tool
def search_tool(query: str) -> str:
    """
    Search the web for real-time information, current events, or facts.
    Returns JSON with synthesized answer and ranked results. (title, url, content snippet, score).
    """
    log.info(" search_tool invoked: %s", query[:100])
    result = _search_web.invoke({"query": query})
    log.info("✅ search_tool complete")
    return result


# ===========================================================================
# 2. EXTRACT  –  /extract
#    Pull full clean text from one or more URLs.
# ===========================================================================
_extractor = TavilyExtract(
    extract_depth="advanced",  
    format="markdown",          
)


@tool
def extract_tool(urls: List[str]) -> str:
    """
    Extract full text content from one or more URLs.
    Returns JSON with results (url, raw_content, images) and failed_results.
    """
    log.info(" extract_tool invoked: %s", urls)
    result = _extractor.invoke({"urls": urls})
    log.info("✅ extract_tool complete")
    return result


# ===========================================================================
# 3. CRAWL  –  /crawl
#    Recursively spider a website and extract content from every page.
# ===========================================================================
_crawler = TavilyCrawl(
    max_depth=2,        # How many link-hops from the start URL (1-10)
    max_breadth=15,     # Max pages to visit per depth level (1-50)
    limit=20,           # Total page cap across the whole crawl (1-1000)
    extract_depth="basic",  # Content extraction quality: "basic" | "advanced"
    allow_external=False,   # Stay on the same domain
    include_images=False,
    include_favicon=False,
    # format="markdown",    # Output format per page
    instructions=None,    # Natural-language guidance for the crawler
    select_paths=None,    # URL path prefixes to include, e.g. ["/docs"]
    exclude_paths=None,   # URL path prefixes to skip, e.g. ["/blog"]
    select_domains=None,  # Extra domains to allow
    exclude_domains=None, # Domains to block
    categories=None,      # Content categories: ["Documentation", "API", …]
)


@tool
def crawl_tool(url: str, instructions: Optional[str] = None) -> str:
    """
    Recursively crawl a website starting from a root URL. 
    Returns JSON list of pages with url and extracted content.
    """
    log.info("crawl_tool invoked: %s", url)
    payload: dict = {"url": url}
    if instructions:
        payload["instructions"] = instructions
    result = _crawler.invoke(payload)
    log.info("✅ crawl_tool complete")
    return result


# ===========================================================================
# 4. MAP  –  /map
#    Generate a site-graph (list of URLs) without extracting content.
#    Cheaper and faster than crawl when you only need the URL structure.
# ===========================================================================
_mapper = TavilyMap(
    max_depth=3,        # How many link-hops from the start URL
    max_breadth=20,     # Max links followed per depth level
    limit=100,          # Total URL cap
    allow_external=False,
    # instructions=None,
    # select_paths=None,
    # exclude_paths=None,
    # select_domains=None,
    # exclude_domains=None,
    # categories=None,
)


@tool
def map_tool(url: str, instructions: Optional[str] = None) -> str:
    """
    Generate a sitemap (list of URLs) without downloading full content.
    Returns JSON with 'base_url' and a 'results' list of discovered URLs.
    """
    log.info(" map_tool invoked: %s", url)
    payload: dict = {"url": url}
    if instructions:
        payload["instructions"] = instructions
    result = _mapper.invoke(payload)
    log.info("✅ map_tool complete")
    return result


# ===========================================================================
# 5. RESEARCH  –  /research  (create task)
#    Submit a long-running deep-research task. Returns a request_id.
#    Use get_research_tool to poll for results.
# ===========================================================================
_researcher = TavilyResearch(
    model="mini",               # "mini" (fast/cheap) | "pro" (thorough)
    citation_format="numbered", # "numbered" | "apa" | "mla" | "chicago" | "harvard"
    stream=False,               # Stream partial results (not supported in @tool)
    # output_schema=None,       # Pydantic model for structured output
)


@tool
def research_tool(task: str, model: str = "mini") -> str:
    """
    Submit a deep research task (async) .Research depth - "mini" (fast) or "pro" (thorough) Returns  JSON with request_id, status, created_at, and the input echo.
    This is a LONG-RUNNING operation,it returns immediately with a 'request_id' and 'status: pending'.Use get_research_tool with the
    returned request_id to retrieve the completed report.Save the request_id and pass it to get_research_tool later.
    """
    log.info(" research_tool invoked (model=%s): %s", model, task[:100])
    result = _researcher.invoke({"input": task, "model": model})
    log.info("✅ research_tool submitted")
    return result


# ===========================================================================
# 6. GET RESEARCH  –  /research  (poll result)
#    Retrieve the completed report for a previously submitted task.
# ===========================================================================
_get_researcher = TavilyGetResearch()


@tool
def get_research_tool(request_id: str) -> str:
    """
    Retrieve results of a research task using request_id.
    Returns JSON with status with value "pending" / "completed" / "failed",
    output (the final report), citations, and metadata.
    """
    log.info(" get_research_tool polling: %s", request_id)
    result = _get_researcher.invoke({"request_id": request_id})
    log.info("✅ get_research_tool complete")
    return result

