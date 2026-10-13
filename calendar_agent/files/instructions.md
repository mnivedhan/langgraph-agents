Build a LangGraph calculator agent in this working directory by following the LangGraph quickstart.

## Step 1: Read the guide

Detect whether this project uses Python or TypeScript/JavaScript. Fetch and follow the matching page; treat it as the source of truth for package names, model strings, and code:

* Python: [https://docs.langchain.com/oss/python/langgraph/quickstart.md](https://docs.langchain.com/oss/python/langgraph/quickstart.md)
* TypeScript: [https://docs.langchain.com/oss/javascript/langgraph/quickstart.md](https://docs.langchain.com/oss/javascript/langgraph/quickstart.md)

Ask the user whether to use the Graph API or the Functional API. If they have no preference, use the Graph API path.

## Step 2: Install dependencies

Install the packages required by the chosen path with the package manager already used in this project.

## Step 3: Configure model credentials

This quickstart uses Anthropic by default. Check whether `ANTHROPIC_API_KEY` is set. If it is not, ask the user to create a key and set it in the shell or a `.env` file, then stop and wait for confirmation. Do not invent, hardcode, or commit API keys. If the user prefers another chat model provider from the integrations docs, adapt the example accordingly after confirming.

## Step 4: Implement the calculator agent

Implement the selected quickstart path end to end: tools for add, multiply, and divide; the agent graph or functional workflow; and a sample invocation that exercises tool calling. Print the final result so the user can verify the run.

## Rules

* Stay scoped to this quickstart. Do not add deployment, evals, or unrelated frameworks unless the user asks.
* Prefer the APIs and structure shown on the fetched guide over inventing a different agent pattern.
* Ask rather than guess when a secret, API choice, or project convention is unclear.