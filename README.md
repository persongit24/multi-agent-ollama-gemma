# multi-agent-ollama-gemma

Multi-Agent System Repo
├── README.md
├── requirements.txt
├── .gitignore
├── .env (store your API keys securely)
└── app.py


# Intelligent Multi-Agent System with Ollama, MCP, Gemma

This repository demonstrates an intelligent multi-agent system built with Python, [Gradio](https://gradio.app/), [praisonaiagents](https://pypi.org/project/praisonaiagents/), and MCP tools. The system integrates multiple specialized AI agents—Airbnb bookings, financial data analysis, web search, a local LLM agent, and a sequential thinking agent.

## Features
- **Airbnb booking agent** using `MCP`.
- **Finance agent** retrieving real-time data with `yfinance`.
- **Web search agent** leveraging DuckDuckGo.
- **Local LLM agent** using `ollama/gemma3:4b`.
- **Sequential thinking agent** with Gemini 2.5 Pro.
- **Context handling** and **URL filtering**.

## Installation

1. **Clone** the repository:
   ```bash
   git clone YOUR_REPO_URL
   cd YOUR_REPO_DIRECTORY
