from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import PIIMiddleware, HumanInTheLoopMiddleware, SummarizationMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import ToolRuntime
from langgraph.store.postgres import PostgresStore
from langgraph.types import Command

load_dotenv()

def get_user_info(runtime: ToolRuntime) -> str:
    """Look up information about the current user."""
    print("Calling tool user info")
    print(runtime.state)
    user = runtime.state["user_name"]
    return user

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to the given address."""
    # replace with real sending logic
    return f"Email sent to {to} with subject '{subject}'"

class UserInfo(AgentState):
    user_name: str

if __name__ == "__main__":
    print("\nUser Preference Bot learns about you and what you like and dislike and suggest you when you are in doubt\n")
    user_name = input("\n What is your name? > ")

    # agent = create_agent(
    #     model="ollama:qwen3:4b",
    #     system_prompt="You are a user preference bot. You learn about the user and their likes and dislikes and recommend the user the best based on their likings",
    #     # tools=[get_user_info],
    #     state_schema=UserInfo,
    #     checkpointer=InMemorySaver(),
    # )

    DB_URI = "postgresql://langchain_agent:password@localhost:5432/langchain_memory?sslmode=disable"
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer, PostgresStore.from_conn_string(DB_URI) as store:
        store.setup()
        checkpointer.setup()  # auto create tables in PostgreSQL
        agent = create_agent(
            # model="ollama:qwen3:4b",
            model="google_genai:gemini-3.8-flash",
            system_prompt="You are a user preference bot. You learn about the user and their likes and dislikes and recommend the user the best based on their likings",
            tools=[get_user_info, send_email],
            state_schema=UserInfo,
            # checkpointer=checkpointer,
            # store=store,
            middleware=[
                PIIMiddleware("email", strategy="redact", apply_to_input=True),
                HumanInTheLoopMiddleware(
                    interrupt_on={
                        # Require approval for sensitive operations
                        "send_email": True,
                        # "delete_database": True,
                        # # Auto-approve safe operations
                        # "search": False,
                    }
                ),
                SummarizationMiddleware(
                    # model="ollama:qwen3:4b",  # model used to write the summary
                    model="google_genai:gemini-3.8-flash",
                    trigger=("tokens", 1500),  # summarize once history exceeds this
                    keep=("messages", 6),  # always keep the last 6 messages as-is
                ),
            ]
        )
        # Run the agent in a loop for interactive conversation
        while True:
            user_input = input("\nYou > ")
            if user_input.lower() == "exit":
                print("Happy conversing!")
                break
            response = agent.invoke(UserInfo(messages=[HumanMessage(content=user_input)], user_name=user_name), config={"configurable": {"thread_id": user_name}}, version="v3")
            # for event in events.messages:
            #     print(f"\n Bot > {event.text}")
            #     print(f"Tokens Used: {event.output.usage_metadata}")

            while "__interrupt__" in response:
                request = response["__interrupt__"][0].value
                for action in request["action_requests"]:
                    print(f"\nApproval needed: {action['name']}({action['args']})")
                choice = input("approve / reject? > ").strip().lower()

                decision = (
                    {"type": "approve"}
                    if choice.startswith("a")
                    else {"type": "reject", "message": "User rejected this action."}
                )
                response = agent.invoke(Command(resume={"decisions": [decision]}), config={"configurable": {"thread_id": user_name}})
            print(response["messages"][-1].text)



"""
PIIMiddleware
pii_type	Type of PII to detect (built-in or custom)	Required
strategy	How to handle detected PII ("block", "redact", "mask", "hash")	"redact"
detector	Custom detector function or regex pattern	None (uses built-in)
apply_to_input	Check user messages before model call	True
apply_to_output	Check AI messages after model call	False
apply_to_tool_results	Check tool result messages after execution	False
"""