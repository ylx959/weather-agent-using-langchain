# Weather Search AI Agent

An AI agent built with LangChain that can search the web and check the live weather for any city. Ask it something like *"Find the capital of Japan and tell me its current weather"*, and it works through the problem step by step using the ReAct pattern (Reason → Act → Observe):

1. Uses **Tavily Search** to look up facts on the web (e.g. "What is the capital of Japan?")
2. Uses **Weatherstack** to fetch the current temperature, conditions and humidity for a city
3. Uses **Google Gemini** to reason over the results and write the final answer

The project ships with two entry points: a command-line script for quick runs and a Streamlit web UI for interactive queries.

## Built With

- Python 3.12
- uv
- LangChain
- Google Gemini
- Tavily Search
- Weatherstack API
- Streamlit

## Getting Started

**1. Install uv**

```bash
brew install uv
```

**2. Install dependencies**

```bash
uv sync
```

**3. Add API keys** — get them from [Google Gemini](https://aistudio.google.com/apikey), [Tavily](https://app.tavily.com) and [Weatherstack](https://weatherstack.com)

```bash
cp .env.example .env
```

```dotenv
GOOGLE_API_KEY=your-google-api-key
TAVILY_API_KEY=your-tavily-api-key
WEATHERSTACK_API_KEY=your-weatherstack-api-key
```

**4. Run**

```bash
uv run main.py                # CLI
uv run streamlit run app.py   # Web UI
```

## Project Structure

```text
weather-search-ai-agent-using-LangChain/
├── README.md          ← You are here
├── app.py             # Streamlit web UI for the agent
├── main.py            # Command-line demo of the agent
├── pyproject.toml     # Project metadata and dependencies (uv)
├── uv.lock            # Locked dependency versions
├── .python-version    # Python version used by uv (3.12)
├── LICENSE            # MIT License
├── .env.example       # Template for the required API keys
├── .env               # Your real API keys (git-ignored)
└── .gitignore
```

## Copyright

© 2026 ylx. Released under the [MIT License](LICENSE).
