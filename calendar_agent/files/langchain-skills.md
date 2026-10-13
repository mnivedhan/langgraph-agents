Install LangChain Skills for my coding agent so it can perform better on LangChain, LangGraph, and Deep Agents tasks.

Use the Agent Skills installer from [https://github.com/langchain-ai/langchain-skills](https://github.com/langchain-ai/langchain-skills):

```bash  theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
npx skills add langchain-ai/langchain-skills --skill '*' --yes
```

If I ask for a global install instead, use:

```bash  theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
npx skills add langchain-ai/langchain-skills --skill '*' --yes --global
```

Detect which agent or editor I am using. If I use Claude Code and prefer the plugin path, follow the marketplace install from that repository README (`/plugin marketplace add` then `/plugin install`). Do not invent alternate skill package names or install URLs. After installing, confirm the skills are available to the agent.
