"""
Standalone terminal test client for mcp_server/server.py.

Spawns the MCP server as a subprocess over stdio (the same transport
deepagents will use), sends a real tools/list request and a real
tools/call request, and prints the raw responses. No deepagents/LangGraph
code involved - this only exercises the MCP protocol layer.
"""

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVER_SCRIPT = Path(__file__).resolve().parent / "server.py"

server_params = StdioServerParameters(
    command="uv",
    args=["run", "--project", str(PROJECT_ROOT), "python", str(SERVER_SCRIPT)],
)


async def main() -> None:
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            init_result = await session.initialize()
            print("=== initialize ===")
            print(f"server: {init_result.serverInfo.name} v{init_result.serverInfo.version}")
            print()

            print("=== tools/list ===")
            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                print(f"- {tool.name}")
                print(f"  description: {tool.description}")
                print(f"  input_schema: {tool.inputSchema}")
            print()

            print("=== tools/call: search_complaints ===")
            call_result = await session.call_tool(
                "search_complaints",
                {
                    "make": "Hyundai",
                    "model": "Kona",
                    "year": 2020,
                    "query": "engine stalling",
                },
            )
            print(f"isError: {call_result.isError}")
            for block in call_result.content:
                print(block.text if hasattr(block, "text") else block)


if __name__ == "__main__":
    asyncio.run(main())
    sys.exit(0)
