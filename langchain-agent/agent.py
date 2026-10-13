from typing import List, Callable
from uuid import uuid7
from langchain.agents import create_agent
import requests
from dotenv import load_dotenv
from langchain.agents.middleware import wrap_tool_call, AgentMiddleware, ModelRequest, wrap_model_call, ModelResponse
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import create_context_cache, ChatGoogleGenerativeAI
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from langgraph.prebuilt import ToolRuntime
from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from rich import print_json

load_dotenv()

WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "fog", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow",
    80: "rain showers", 81: "rain showers", 82: "heavy rain showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail",
}

##Tool 1
@tool
def get_weather(city: str):
    """Get the current weather for a city. Pass only the city name, e.g. 'San Francisco'."""
    try:
        # Step 1: city name -> coordinates
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10,
        ).json()
        if not geo.get("results"):
            return f"Could not find a city named '{city}'."
        place = geo["results"][0]

        # Step 2: coordinates -> current weather
        wx = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,wind_speed_10m,weather_code",
            },
            timeout=10,
        ).json()["current"]

        condition = WEATHER_CODES.get(wx["weather_code"], "unknown conditions")
        return (
            f"{place['name']}, {place.get('country', '')}: "
            f"{wx['temperature_2m']}°C, {condition}, wind {wx['wind_speed_10m']} km/h."
        )
    except Exception as e:
        return f"Weather lookup failed: {e}"

##Tool 2
@tool
def calculator(a: int, b: int):
    """Sums up the given two integers"""
    return a + b

@tool
def user_city(name: str):
    """"Gives the current city of the user"""
    return f"Current city of user : {name} is delhi"

@tool
def change_state_to_city(city: str, runtime: ToolRuntime) -> Command:
    """Change the current state to the specified city."""
    return Command(
        update={
            "current_city": city,
            "messages": [
                ToolMessage(
                    content=f"State changed to {city}.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )

@wrap_tool_call
def tools_exception_handler(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage],
):
    try:
        return handler(request)
    except Exception as exe:
        return ToolMessage(content=f"Error while executing request: {request}", tool_call_id=request.tool_call["id"])


class Answer(BaseModel):
    weathers: List[int]
    total: float


tools = [user_city, get_weather, calculator]
print([tool.name for tool in tools])


config = {"configurable": {"thread_id": str(uuid7())}}

@dataclass
class RequestContext:
    user_id: str

model = init_chat_model(
    model="google_genai:gemini-3.6-flash",
    thinking_budget=-1,
    config=config
)

# print(model.profile)
#
# model.bind_tools([get_weather])


# large_content = "Your long document, system instructions, or data text here..." * 80
# cache = create_context_cache(
#     model=model,
#     messages=[
#         SystemMessage(content="You are a helpful assistant."),
#         HumanMessage(content=large_content),
#     ],
#     tools=[user_city],
#     ttl="3600s"
# )
# print(cache)

##Dyanmically selecting tools
##option 1 (state based, store based, context based selection of tools)

@wrap_model_call
def store_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Filter tools based on Store preferences."""
    user_id = request.runtime.context.user_id

    # Read from Store: get user's enabled features
    store = request.runtime.store
    feature_flags = store.get(("features",), user_id)

    if feature_flags:
        enabled_features = feature_flags.value.get("enabled_tools", [])
        # Only include tools that are enabled for this user
        tools = [t for t in request.tools if t.name in enabled_features]
        request = request.override(tools=tools)

    return handler(request)

##Option 2 (Dynamically adding tools)
@tool
def calculate_tip(bill_amount: float, tip_percentage: float = 20.0) -> str:
    """Calculate the tip amount for a bill."""
    tip = bill_amount * (tip_percentage / 100)
    return f"Tip: ${tip:.2f}, Total: ${bill_amount + tip:.2f}"

class DynamicToolMiddleware(AgentMiddleware):
    """Middleware that registers and handles dynamic tools."""

    def wrap_model_call(self, request: ModelRequest, handler):
        # Add dynamic tool to the request
        # This could be loaded from an MCP server, database, etc.
        updated = request.override(tools=[*request.tools, calculate_tip])
        return handler(updated)

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        # Handle execution of the dynamic tool
        if request.tool_call["name"] == "calculate_tip":
            return handler(request.override(tool=calculate_tip))
        return handler(request)

basic_agent = create_agent(
    # model="anthropic:claude-sonnet-5-5",
    # model="google_genai:gemini-3.8-flash",
    model=model,
    tools=[user_city, get_weather, calculator, change_state_to_city], # tool calling
    system_prompt="You are a helpful assistant",
    response_format=Answer,  # Structured Output
    context_schema = RequestContext,
    middleware=[tools_exception_handler, store_based_tools, DynamicToolMiddleware()]

)

result = basic_agent.invoke(
    {"messages": [{"role": "user", "content": "Get the weather of tamilnadu"}]},
    config=config,  # Scoping the conversation together
    context=RequestContext(user_id="Nivedhan")
)
print(result)

# print(basic_agent)
# ##Content -> Get the weather of delhi and paris and calculate their total weather" Sequential tool calls, first call get_weather(delhi) then get_weather(paries), then calculator(20+32)
# result = basic_agent.invoke(
#     {"messages": [{"role": "user", "content": "Get the weather of delhi and paris and calculate their total weather"}]},
#     config=config,  # Scoping the conversation together
#     context=RequestContext(user_id="Nivedhan")
# )
# print(result)

# stream = basic_agent.stream_events(
#     # [HumanMessage(content=[{"type": "text", "text": "Search for AI news and summarize the findings"}])],
#     {"messages": [{"role": "user", "content": [{"type": "text", "text": "Get the weather of delhi and paris and calculate their total weather"}]}]},
#     version="v3",
# )
# for snapshot in stream.values:
#     # Each snapshot contains the full state at that point
#     print(snapshot)
#     latest_message = snapshot["messages"][-1]
#     if latest_message.content:
#         if isinstance(latest_message, HumanMessage):
#             print(f"User: {latest_message.content}")
#         elif isinstance(latest_message, AIMessage):
#             print(f"Agent: {latest_message.content}")
#     elif latest_message.tool_calls:
#         print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")

## Batch the input
# results = basic_agent.batch([
#     {
#         "messages": [
#             {"role": "user", "content": "What is the weather in Delhi?"}
#         ]
#     },
#     {
#         "messages": [
#             {"role": "user", "content": "What is the weather in Paris?"}
#         ]
#     },
#     {
#         "messages": [
#             {"role": "user", "content": "What is the total weather?"}
#         ]
#     },
# ])
# for result in results:
#     print(result)








"""
{'messages': [
    HumanMessage(
        content="What's 2 + 3, and what's the weather in San Francisco?",
        additional_kwargs={},
        response_metadata={},
        id='e35a6db5-7bd7-4585-86d2-72c428d39701'
    ),
    AIMessage(
        content=[],
        additional_kwargs={'function_call': {'name': 'get_weather', 'arguments': '{"state": "San Francisco"}'}, '__gemini_function_call_thought_signatures__': {'call_853451': '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'}},
        response_metadata={'finish_reason': 'STOP', 'model_name': 'gemini-3.6-flash', 'safety_ratings': [], 'model_provider': 'google_genai'},
        id='lc_run--01a0f12f-eda0-7e92-b211-8b05e9a8f1d5-0',
        tool_calls=[{'name': 'calculator', 'args': {'a': 2, 'b': 3}, 'id': 'call_853451', 'type': 'tool_call'}, {'name': 'get_weather', 'args': {'state': 'San Francisco'}, 'id': 'call_853452', 'type': 'tool_call'}],
        invalid_tool_calls=[],
        usage_metadata={
            'input_tokens': 122,
            'output_tokens': 565,
            'total_tokens': 687,
            'input_token_details': {'cache_read': 0}, 
            'output_token_details': {'reasoning': 532}}
    ),
    ToolMessage(
        content='5',
        name='calculator',
        id='2bfdf389-95f5-4122-a282-3f9c9b836386',
        tool_call_id='call_853451'
    ),
    ToolMessage(
        content="It's always sunny in San Francisco!",
        name='get_weather',
        id='45817067-097b-43df-a9da-e72520ae0884',
        tool_call_id='call_853452'
    ),
    AIMessage(
        content=[{'type': 'text', 'text': '2 + 3 is **5**.\n\nAs for San Francisco, the weather is currently sunny!', 'extras': {'signature': 'EpQCCpECAWkUfRMbaUNlKz3hwa7zJEiv0RyyCTWQdX2iHoO+/tc6EYdqjIxHE9Vl+JWc393Nvh45BMSO9H6UfIa4Fki9eTn1ccR0Xz6PHI3o5/K4jeDVpN+Kxu31OS6a/2bbvjDBch3meuRZ/5ISgvSdYHIx4FR0zTMLTZyMzdrURv3RUh4IaNSZvTM/NBd13SAoGsZaylQsUzmeUyoAFuxYyrKd6AyAY57OeLTzPHKtYKlm3vM9LLjbKAvGurojM1MOS6emmALuwAIUBmxXNr0vLCn+KgD+lV2AT+Or4AigkoSsbnwBtWQcUuNwIebS5XiV3JhcC713vtEVBN/IOtDz6nzkw8LD0NQe+03KsGIhCqn1tOau'}}],
        additional_kwargs={},
        response_metadata={'finish_reason': 'STOP', 'model_name': 'gemini-3.6-flash', 'safety_ratings': [], 'model_provider': 'google_genai'},
        id='lc_run--01a0f130-01e1-71a0-9fe2-aa897a8b37fd-0',
        tool_calls=[],
        invalid_tool_calls=[],
        usage_metadata={
            'input_tokens': 717,
            'output_tokens': 65,
            'total_tokens': 782,
            'input_token_details': {'cache_read': 0},
            'output_token_details': {'reasoning': 45}}
    )
]}
"""
