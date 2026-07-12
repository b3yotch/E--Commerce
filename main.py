"""
Standalone runner for the Product Research Agent.

Usage:
    python main.py https://example-store.com/products/widget

Prints the validated ProductResearch JSON to stdout. This exists so the
agent can be tested in isolation before the rest of the pipeline (Creative
Strategy, Prompt Generation, etc.) exists to consume its output.
"""

from __future__ import annotations

import asyncio
import json
import sys

from app.agents.research_agent.graph import research_graph


async def run(url: str) -> None:
    result = await research_graph.ainvoke({"url": url, "retries": 0})

    if result.get("error"):
        print(f"FAILED: {result['error']}", file=sys.stderr)
        sys.exit(1)

    research = result["research"]
    print(research.model_dump_json(indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <product_page_url>", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run(sys.argv[1]))
