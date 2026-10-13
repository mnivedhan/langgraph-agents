from langchain.agents import AgentState, create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
from langgraph.prebuilt import ToolRuntime
from pydantic import BaseModel

load_dotenv()

@tool("weather", description="For given city, it will return weather")
def get_weather(city: str):
    return f"It is sunny in {city}"


agent = create_agent(
    model="google_genai:gemini-3.6-flash",
    tools=[get_weather],
    system_prompt="You are an weather Agent",
)

stream = agent.stream_events(
    AgentState(
        messages=[HumanMessage(content="what's the weather in San Francisco?")],
    ),
    version="v3"
)

#stream all events
# for event in stream:
#     print(event)

# stream only delta messages
for message in stream.messages:
    for delta in message.text:
        print(delta, end="", flush=True)



# final_state = stream.output
#
# print(f"final_state: {final_state}")