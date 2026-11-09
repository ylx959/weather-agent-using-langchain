import os
from pathlib import Path
import requests
import streamlit as st
import certifi
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_classic.agents import (
    create_react_agent,
    AgentExecutor
)

from langchain.agents import create_agent

from langchain_core.prompts import PromptTemplate

from langchain_tavily import TavilySearch

# ==========================================
# LOAD ENV VARIABLES
# ==========================================
os.environ["SSL_CERT_FILE"] = certifi.where()
# .env lives in the project root, one level above code/
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ==========================================
# STREAMLIT PAGE CONFIG
# ==========================================


st.set_page_config(
    page_title="Weather Assistant",
    page_icon="🌤️",
    layout="centered"
)

st.title("🌤️ Weather Assistant")
st.markdown("Search + Weather AI Agent using LangChain")

# ==========================================
# SEARCH TOOL
# ==========================================

search_tool = TavilySearch(max_results=2)

# ==========================================
# WEATHER TOOL
# ==========================================

@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """

    url = (
        f"http://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )

    response = requests.get(url)

    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}"

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )


# ==========================================
# LLM
# ==========================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
)

# ==========================================
# PROMPT
# ==========================================

REACT_TEMPLATE = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(REACT_TEMPLATE)

# ==========================================
# TOOLS
# ==========================================

tools = [
    search_tool,
    get_weather_data
]

# ==========================================
# CREATE AGENT
# ==========================================

agent = create_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# ==========================================
# EXECUTOR
# ==========================================

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# ==========================================
# UI INPUT
# ==========================================

def to_text(output) -> str:
    """Gemini may return a list of content parts instead of a plain string."""
    if isinstance(output, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in output
        )
    return str(output)


# st.form: pressing Enter in the text box submits the form
with st.form("query_form"):
    user_query = st.text_input(
        "Enter your query:",
        placeholder="Example: Find the capital of India and current weather"
    )
    submitted = st.form_submit_button("Run Agent")

# ==========================================
# RUN AGENT
# ==========================================

if submitted:

    if user_query:

        with st.spinner("Agent is thinking..."):

            try:
                response = agent_executor.invoke({
                    "input": user_query
                })
                # keep the answer across Streamlit reruns
                st.session_state["answer"] = to_text(response["output"])
                st.session_state.pop("error", None)

            except Exception as e:
                st.session_state["error"] = str(e)
                st.session_state.pop("answer", None)

    else:
        st.warning("Please enter a query")

if "error" in st.session_state:
    st.error(f"Error: {st.session_state['error']}")

if "answer" in st.session_state:
    st.success("Response Generated")
    st.markdown("## Final Response")
    st.write(st.session_state["answer"])
