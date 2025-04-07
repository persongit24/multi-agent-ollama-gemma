import os
import re
import gradio as gr
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

from praisonaiagents import Agent, MCP
from praisonaiagents.tools import duckduckgo, get_stock_price, get_stock_info, get_historical_data

def remove_urls(text):
    """Remove URLs from text using regex"""
    url_pattern = re.compile(r'https?://\\S+|www\\.\\S+')
    return url_pattern.sub('', text)

# Airbnb Agent
def airbnb_agent(query, context=None):
    agent = Agent(
        instructions="Handle Airbnb bookings without providing URLs.",
        llm="ollama/gemma3:4b",
        tools=MCP("npx -y @openbnb/mcp-server-airbnb --no-urls")
    )
    prompt = f"{context}\\n{query}" if context else query
    return remove_urls(agent.start(prompt))

# Finance Agent
def finance_agent(query, context=None):
    agent = Agent(
        instructions="Analyze financial data without providing URLs.",
        llm="ollama/gemma3:4b",
        tools=[get_stock_price, get_stock_info, get_historical_data]
    )
    prompt = f"{context}\\n{query}" if context else query
    return remove_urls(agent.start(prompt))

# Web Search Agent
def web_search_agent(query, context=None):
    agent = Agent(
        instructions="Perform web research and summarize findings without URLs.",
        llm="ollama/gemma3:4b",
        tools=[duckduckgo]
    )
    prompt = f"{context}\\n{query}" if context else query
    result = agent.start(prompt)
    if result:
        return remove_urls(f"Web results for '{query}':\\n{result}")
    else:
        return "No web results found"

# Local LLM Agent
def local_llm_agent(query, context=None):
    agent = Agent(
        instructions="Answer general queries using local LLM without URLs.",
        llm="ollama/gemma3:4b"
    )
    prompt = f"{context}\\n{query}" if context else query
    response = agent.start(prompt)
    return remove_urls(response) if response else "I couldn't generate a response for that query"

# Sequential Thinking Agent (Gemini 2.x Thinking)
def sequential_agent(query, context=None):
    agent = Agent(
        instructions="Break down complex problems into step-by-step processes without URLs.",
        llm="gemini/gemini-2.0-flash-thinking-exp-01-21",
        tools=MCP(
            "npx -y @modelcontextprotocol/server-sequential-thinking",
            env={"GOOGLE_API_KEY": google_api_key}
        )
    )
    prompt = f"{context}\\n{query}" if context else query
    return remove_urls(agent.start(prompt))

# Enhanced Web Search with Time Awareness
def time_aware_web_search(query, context=None):
    requires_recent = any([
        "recent" in query.lower(),
        "latest" in query.lower(),
        "current" in query.lower(),
        str(datetime.now().year) in query  # e.g., '2023'
    ])

    if requires_recent:
        web_result = web_search_agent(query, context)
        if web_result and isinstance(web_result, str) and len(web_result.strip()) > 0:
            return web_result

    # Fallback to local LLM
    return local_llm_agent(query, context)

# Routing Logic
def route_query(query, chat_history):
    try:
        query_lower = query.lower()
        # Use the previous user message as context, if available
        context = chat_history[-1][0] if chat_history else None

        if any(kw in query_lower for kw in ["think", "step-by-step", "process", "break down"]):
            return f\"🧠 Sequential Agent (Gemini 2.x Thinking)\\n\\n{sequential_agent(query, context)}\"
        elif any(kw in query_lower for kw in ["airbnb", "book", "apartment", "stay"]):
            return f\"🏠 Airbnb Agent\\n\\n{airbnb_agent(query, context)}\"
        elif any(kw in query_lower for kw in ["stock", "finance", "price", "tsla", "nasdaq"]):
            return f\"📈 Finance Agent\\n\\n{finance_agent(query, context)}\"
        elif any(kw in query_lower for kw in ["search", "find", "lookup", "google"]):
            return f\"🔍 Web Search Agent\\n\\n{web_search_agent(query, context)}\"
        else:
            return f\"🤖 Smart Assistant\\n\\n{time_aware_web_search(query, context)}\"
    except Exception as e:
        return f\"❌ Error\\n\\n{str(e)}\"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown(\"## Intelligent Multi-Agent System\")

    chatbot = gr.Chatbot(label=\"Ask me anything!\", show_label=False)

    with gr.Row():
        msg = gr.Textbox(
            scale=4,
            placeholder=\"Enter your request...\",
            show_label=False,
            container=False
        )
        btn = gr.Button(\"Submit\", variant=\"primary\")

    def respond(message, chat_history):
        if not message.strip():
            return \"Please enter a valid query\", chat_history

        response = route_query(message, chat_history)
        if not isinstance(response, str) or not response.strip():
            response = \"No valid response generated\"

        chat_history.append((message, response))
        return \"\", chat_history

    btn.click(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == \"__main__\":
    demo.launch()
