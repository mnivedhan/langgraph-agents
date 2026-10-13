import asyncio
import uuid

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.mcp import MCPAdapter
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

load_dotenv()

model = init_chat_model(model="google_genai:gemini-3.6-flash")

@tool
def sum(a: int, b: int):
    """Sum two numbers"""
    return a + b

print(sum) #name='sum' description='Sum two numbers'
# args_schema=<class 'langchain_core.utils.pydantic.sum'>
# func=<function sum at 0x10def6cf0>

async def get_tools():
    async with MCPAdapter("http://localhost:8000/mcp") as adapter:
        tools = await adapter.list_tools()
        return tools

async def run_agent():
    # MCP Adapter is async so use agent.ainvoke and await
    async with MCPAdapter("http://localhost:8000/mcp") as adapter:
        config = {"configurable": {"thread_id": str(uuid.uuid7())}}
        tools = await adapter.list_tools()
        print(tools[0]) #name='add' description='Add two numbers and return the sum.'
        # args_schema= {'type': 'object', 'additionalProperties': False, 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}}, 'required': ['a', 'b']}
        # metadata={'mcp': {'tool': {'_meta': {'fastmcp': {'tags': []}}}, 'server': {'name': 'Calculator Tools', 'version': '4.0.10'}}}
        # handle_tool_error=<function _handle_mcp_tool_error at 0x10f6e1010>
        # response_format='content_and_artifact'
        # coroutine=<function as_langchain_tool.<locals>.call_tool at 0x111a5c5c0>
        agent = create_agent(model=model, tools=tools, system_prompt="You are an MCP expert")
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "calculate sum of 2 and 3"}]}, config=config)
        print(result)


if __name__ == '__main__':
    asyncio.run(run_agent())

    # tools = asyncio.run(get_tools())
    # model.bind_tools(tools)
    # result = model.invoke([HumanMessage("calculate sum of 2 and 3")])
    # print(result)