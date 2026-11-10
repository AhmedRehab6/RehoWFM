import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from tools import analysis_tools

# 1. Force Python to prioritize the .env file over any cached terminal variables
load_dotenv(override=True)

choice = input("Which model? (1 = Claude, 2 = Gemini): ").strip()
USE_GEMINI = choice == "2"

class AnalysisResponse(BaseModel):
    question: str
    findings: str
    assumptions: list[str]
    recommendations: list[str]
    files_created: list[str]
    tools_used: list[str]

llm_claude = ChatAnthropic(
    model="claude-sonnet-5-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

llm_gemini = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

llm = llm_gemini if USE_GEMINI else llm_claude
print("Using model:", "Gemini" if USE_GEMINI else "Claude")

parser = PydanticOutputParser(pydantic_object=AnalysisResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a senior data analyst and workforce management (WFM) expert covering
            forecasting, intraday planning, staffing and long-term capacity planning for
            contact centers and operations, plus Excel and Power BI analysis.

            Working method:
            1. Files are in the data folder (results in output). List and inspect a file
               before using it. Check date ranges, gaps, duplicates and outliers.
            2. For forecasts, use the forecast tool, which backtests several methods.
               Report the chosen method, WAPE and bias, and the forecast range. Never
               present a forecast without its accuracy. Flag holidays, campaigns or
               outages that the data cannot explain.
            3. For staffing, use Erlang C tools, never mental math. Convert to scheduled
               FTE using shrinkage. For capacity plans use the capacity plan tool and
               state the shortfall or surplus and hires needed.
            4. Use these defaults only when the user gives no value, and always list them
               under assumptions: service level 80 percent in 20 seconds, occupancy cap
               85 percent, shrinkage 30 percent, 30-minute intervals, 40 paid hours per
               week, 300 seconds AHT, annual attrition 30 percent.
            5. Ask for missing critical inputs (AHT, SL target, headcount) only if a
               sensible default would mislead. Otherwise proceed and state assumptions.
            6. Never invent numbers. Every figure must come from a tool result. Mention
               limits: Erlang C ignores abandonment and multi-skill pooling.
            7. Give clear, practical recommendations a WFM manager can act on.
            8. Write DAX or Power Query M only when asked for Power BI help.

            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = analysis_tools
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def extract_text(output) -> str:
    """Claude may return a list of content blocks, Gemini usually a plain string."""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts = []
        for block in output:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(output)


chat_history = []
while True:
    query = input("\nWhat would you like to analyze? (press Enter on an empty line to quit) ").strip()
    if not query:
        break
    raw_response = agent_executor.invoke({"query": query, "chat_history": chat_history})
    text = extract_text(raw_response.get("output"))
    try:
        print(parser.parse(text))
    except Exception as e:
        print("Error parsing response", e, "Raw Response - ", text)
    chat_history.append(("human", query))
    chat_history.append(("ai", text))