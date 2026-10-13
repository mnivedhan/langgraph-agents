from langchain.agents import AgentState, create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
from langgraph.prebuilt import ToolRuntime
from pydantic import BaseModel

load_dotenv()


@tool("weather", description="For given city, it will return weather", return_direct=True)
def get_weather(city: str, runtime: ToolRuntime):
    return f"Tell {runtime.state.get("user_name")} and {runtime.context.user_id},  It is sunny in {city}"
    # To update user_id
    return Command(update={
            "user_name": "Nivedhan Murugavel",
            # update the message history
            "messages": [
                ToolMessage(
                    "Successfully looked up user information",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        })

class CustomAgentState(AgentState):
    user_name: str
    preferences: dict

class CustomContext(BaseModel):
    user_id: str

agent = create_agent(
    model="google_genai:gemini-3.6-flash",
    tools=[get_weather],
    system_prompt="You are an weather Agent",
    state_schema=CustomAgentState,
    context_schema=CustomContext
)

agent_responses = agent.invoke(
    CustomAgentState(
        messages=[HumanMessage(content="what's the weather in San Francisco?")],
        user_name ="Nivedhan",
        preferences = {}
    ),
    context=CustomContext(user_id="Vithya")
)

print(agent_responses)