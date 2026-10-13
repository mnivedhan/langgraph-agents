from langchain.chat_models import init_chat_model
from tools import add, multiply, divide

model = init_chat_model("claude-sonnet-4-6", temperature=0)

tools = [add, multiply, divide]
tools_by_name = {t.name: t for t in tools}

print(tools_by_name)
model_with_tools = model.bind_tools(tools)