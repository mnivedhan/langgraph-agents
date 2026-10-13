from sys import thread_info

from jaraco.functools import first_invoke
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from dotenv import load_dotenv

from user_preference_agent.agent import UserInfo

load_dotenv()

def get_user_info() -> str:
    """Look up information about the current user."""
    return "No user profile on file."

if __name__ == "__main__":
    print("\nBot learns about likes and dislikes and suggest you when you are in doubt\n")
    user_name = input("\n What is your name? > ")

    DB_URI = "postgresql://langchain_agent:password@localhost:5432/langchain_memory?sslmode=disable"
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()  # auto create tables in PostgreSQL
        agent = create_agent(
            model="google_genai:gemini-3.6-flash",
            system_prompt="You are a user preference bot. You learn about the user likes and dislikes and recommend the user the best based on his likings",
            tools=[get_user_info],
            checkpointer=checkpointer,
        )
        # Run the agent in a loop for interactive conversation
        while True:
            user_input = input("\nYou > ")
            if user_input.lower() == "exit":
                print("Happy conversing!")
                break
            response = agent.invoke({"messages": [{"role": "user", "content": user_input}]}, config={"configurable": {"thread_id": user_name}})
            print(f"\nMedical Bot > {response["messages"][-1].text}")