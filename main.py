# hf_trending_mcp.py
import asyncio
from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from typing import AsyncIterator, List, Dict, Any
from datetime import datetime
import httpx

# Base URL for Hugging Face API
HF_API_BASE = "https://huggingface.co/api"

# Define the application context type
class AppContext:
    def __init__(self):
        self.http_client = httpx.AsyncClient(
            headers={"User-Agent": "hf-trending-mcp/1.0"}
        )

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with HTTP client initialization"""
    app_context = AppContext()
    try:
        # Startup: Initialize resources
        await app_context.http_client.__aenter__()
        yield app_context
    finally:
        # Shutdown: Cleanup resources
        await app_context.http_client.__aexit__(None, None, None)

# Initialize MCP server with lifespan
mcp = FastMCP("HF-Trending", dependencies=["httpx"], lifespan=app_lifespan)

async def fetch_trending(endpoint: str, limit: int, ctx: Context) -> List[Dict[str, Any]]:
    """Helper function to fetch trending items from HF API"""
    client = ctx.request_context.lifespan_context.http_client
    try:
        response = await client.get(
            f"{HF_API_BASE}/{endpoint}",
            params={"sort": "trendingScore", "limit": limit}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return [{"error": f"Failed to fetch trending {endpoint}: {str(e)}"}]

# Tools for trending content
@mcp.tool()
async def get_trending_models(limit: int = 10, ctx: Context = None) -> str:
    """
    Get trending models from Hugging Face.
    
    Parameters:
        limit (int): Number of trending models to return (default: 10)
        ctx (Context): MCP context providing access to lifespan resources (automatically injected)
    """
    models = await fetch_trending("models", limit, ctx)
    return "\n".join([
        f"{m['id']} (Downloads: {m.get('downloads', 0)}, Likes: {m.get('likes', 0)})\nTags: {', '.join(m['tags'])}\n"
        for m in models
    ])

@mcp.tool()
async def get_trending_datasets(limit: int = 10, ctx: Context = None) -> str:
    """
    Get trending datasets from Hugging Face.
    
    Parameters:
        limit (int): Number of trending datasets to return (default: 10)
        ctx (Context): MCP context providing access to lifespan resources (automatically injected)
    """
    datasets = await fetch_trending("datasets", limit, ctx)
    return "\n".join([
        f"{d['id']} (Downloads: {d.get('downloads', 0)}, Likes: {d.get('likes', 0)})\nTags: {', '.join(d['tags'])}\n"
        for d in datasets
    ])

@mcp.tool()
async def get_trending_spaces(limit: int = 10, ctx: Context = None) -> str:
    """
    Get trending spaces from Hugging Face.
    
    Parameters:
        limit (int): Number of trending spaces to return (default: 10)
        ctx (Context): MCP context providing access to lifespan resources (automatically injected)
    """
    spaces = await fetch_trending("spaces", limit, ctx)
    return "\n".join([
        f"{s['id']} (Likes: {s.get('likes', 0)}, SDK: {s.get('sdk', 'N/A')})\nTags: {', '.join(s['tags'])}\n"
        for s in spaces
    ])

@mcp.tool()
async def search_trending(query: str, type: str = "models", limit: int = 10, ctx: Context = None) -> str:
    """
    Search trending items on Hugging Face with a query.
    
    Parameters:
        query (str): Search term to filter trending items
        type (str): Type of items to search ('models', 'datasets', or 'spaces', default: 'models')
        limit (int): Number of results to return (default: 5)
        ctx (Context): MCP context providing access to lifespan resources (automatically injected)
    """
    valid_types = ["models", "datasets", "spaces"]
    if type not in valid_types:
        return f"Invalid type. Must be one of: {', '.join(valid_types)}"
    
    client = ctx.request_context.lifespan_context.http_client
    try:
        response = await client.get(
            f"{HF_API_BASE}/{type}",
            params={"search": query, "sort": "trendingScore", "limit": limit}
        )
        response.raise_for_status()
        items = response.json()
        return "\n".join([
          f"{item['id']} (Likes: {item.get('likes', 0)})\nTags: {', '.join(item['tags'])}\n" 
          for item in items
        ])
    except Exception as e:
        return f"Error searching trending {type}: {str(e)}"

# Prompt for trending analysis
@mcp.prompt()
def analyze_trends() -> str:
    """Analyze current trending items on Hugging Face"""
    return (
        "Please analyze the current trending items on Hugging Face:\n"
        "1. Fetch the top trending models using the get_trending_models tool\n"
        "2. Fetch the top trending datasets using the get_trending_datasets tool\n"
        "3. Fetch the top trending spaces using the get_trending_spaces tool\n"
        "4. Provide a summary of what's currently popular and why you think that might be"
    )

# Run the server
if __name__ == "__main__":
    mcp.run()
