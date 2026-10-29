"""Lab 4 (Week 2, Day 4): a Deep Research agent team.

Give it a question; it plans web searches, runs them in parallel, writes a long report
from the results, and emails the report to you. One of the classic agentic use cases.

Built the reliable way: orchestrated by code (a separate Runner.run() per step), with
structured outputs wherever one agent's output feeds the next. Four agents:
  1. Planner Agent - turns the question into HOW_MANY_SEARCHES web searches
  2. Search Agent  - runs one search with OpenAI's hosted WebSearchTool and summarises it
  3. Writer Agent  - writes a detailed markdown report from all the summaries
  4. Email Agent   - turns the report into a well presented HTML email and sends it

Cost note: WebSearchTool is a paid hosted tool, about 1 cent per search (sometimes
more), so a run costs around 5 cents in searches plus the model calls. Prices:
https://developers.openai.com/api/docs/pricing

Setup:
  - OPENAI_API_KEY in .env.
  - Email settings as in Lab 2 (EMAIL_ADDRESS, EMAIL_SMTP_SERVER, EMAIL_APP_PASSWORD), or
    set USE_EMAIL = False below and use Pushover (PUSHOVER_USER, PUSHOVER_TOKEN).
  - deep_research/ holds the same agents as separate modules with a Gradio UI.

Run: python lab4_deep_research.py ["your research question"]
Then look at the trace: https://platform.openai.com/traces
"""

import asyncio
import sys

from agents import Agent, Runner, WebSearchTool, function_tool, trace
from agents.model_settings import ModelSettings
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from messenger import push, send_email

load_dotenv(override=True)

# ---- Settings ---------------------------------------------------------------------

MODEL_NAME = "gpt-5.4-mini"
USE_EMAIL = True                        # False: send the report as a push notification instead
HOW_MANY_SEARCHES = 5
DEFAULT_QUERY = "Most popular AI Agent frameworks in 2026"


# ---- Agent 1: the Planner Agent ---------------------------------------------------
# Structured outputs: the Field descriptions tell the model what to put in each field.

class WebSearchItem(BaseModel):
    """One planned search: the term, and why it helps answer the question."""

    reason: str = Field(description="Your reasoning for why this search is important to the query.")
    query: str = Field(description="The search term to use for the web search.")


class WebSearchPlan(BaseModel):
    """The planner's output: the list of searches to run."""

    searches: list[WebSearchItem] = Field(description="A list of web searches to perform to best answer the query.")


planner_agent = Agent(
    name="Planner Agent",
    instructions=f"""
You are a research assistant. Given a user query, come up with a set of web searches
to perform to best answer the query. Output {HOW_MANY_SEARCHES} terms to query for.
""",
    model=MODEL_NAME,
    output_type=WebSearchPlan,
)


# ---- Agent 2: the Search Agent ----------------------------------------------------
# WebSearchTool is one of OpenAI's hosted tools: it runs on OpenAI's servers (others
# include FileSearchTool, CodeInterpreterTool and ImageGenerationTool). Quick to use,
# but paid, and only works with OpenAI models.

search_agent = Agent(
    name="Search Agent",
    instructions="""
You are a research assistant. Given a search term, you search the web for that term and
produce a concise summary of the results. The summary must 2-3 paragraphs and less than 300 words.
Capture the main points and be succinct. Reply only with the summary.
""",
    tools=[WebSearchTool()],
    model=MODEL_NAME,
    model_settings=ModelSettings(tool_choice="required"),   # it must search, not answer from memory
)


# ---- Agent 3: the Writer Agent ----------------------------------------------------

class ReportData(BaseModel):
    """The writer's output: a summary, the full report, and ideas for further research."""

    short_summary: str = Field(description="A short 2-3 sentence summary of the findings.")
    markdown_report: str = Field(description="The final report")
    follow_up_questions: list[str] = Field(description="Suggested topics to research further")


writer_agent = Agent(
    name="Writer Agent",
    instructions="""
You are a senior researcher tasked with writing a cohesive report for a research query.
You will be provided with the original query, and some research.
Generate a comprehensive report based on the research and the query.
The final output should be in markdown format, and it should be lengthy and detailed. Aim
for 5-10 pages of content, at least 1000 words.
""",
    model=MODEL_NAME,
    output_type=ReportData,
)


# ---- Agent 4: the Email Agent -----------------------------------------------------

@function_tool
def send_email_tool(subject: str, text_body: str, html_body: str) -> str:
    """
    Send out an email with the given subject and body to all sales prospects

    Args:
        subject: The subject of the email
        text_body: The body of the email as plain text
        html_body: The HTML body of the email
    """
    if USE_EMAIL:
        send_email(subject, text_body, html_body)
    else:
        push(f"Subject: {subject}\n\n{text_body}")
    return "Email sent successfully"


email_agent = Agent(
    name="Email Agent",
    instructions="""
You are provided with a detailed report. Use your tool to send an email, converting the report into
a clean, well presented HTML email with an appropriate subject line.
""",
    tools=[send_email_tool],
    model=MODEL_NAME,
)


# ---- Orchestration by code: one function per step ------------------------------------

async def plan_searches(query: str) -> list[WebSearchItem]:
    """Ask the planner which searches to run for the query."""
    print("Planning searches...")
    result = await Runner.run(planner_agent, f"Query: {query}")
    searches = result.final_output.searches
    for item in searches:
        print(f"  - {item.query}  (why: {item.reason})")
    return searches


async def search(item: WebSearchItem) -> str:
    """Run one planned search and return the Search Agent's summary of the results."""
    input_message = f"Search term: {item.query}\nReason for searching: {item.reason}"
    result = await Runner.run(search_agent, input_message)
    return result.final_output


async def run_searches(query: str) -> list[str]:
    """Plan the searches, then run them all at the same time; return their summaries."""
    searches = await plan_searches(query)
    print(f"Will perform {len(searches)} searches")
    results = await asyncio.gather(*(search(item) for item in searches))
    print("Finished searching")
    return results


async def write_report(query: str, search_results: list[str]) -> ReportData:
    """Have the Writer Agent turn the query and the search summaries into a report."""
    print("Thinking about report...")
    input_message = f"Original query: {query}\nSummarized search results: {search_results}"
    result = await Runner.run(writer_agent, input_message)
    print("Finished writing report")
    return result.final_output


async def send_report_email(report: ReportData):
    """Have the Email Agent format the report as an HTML email and send it."""
    print("Writing email...")
    result = await Runner.run(email_agent, report.markdown_report)
    print("Email sent")
    return result.final_output


# ---- Main -------------------------------------------------------------------------

async def main():
    """Research the question given on the command line (or DEFAULT_QUERY), all in one trace."""
    query = " ".join(sys.argv[1:]) or DEFAULT_QUERY
    with trace("Research trace"):
        print(f"Starting research on: {query}")
        search_results = await run_searches(query)
        report = await write_report(query, search_results)
        await send_report_email(report)
        print("Hooray!")

    print(f"\n{'=' * 10} Summary {'=' * 10}\n\n{report.short_summary}")
    print(f"\n{'=' * 10} Report {'=' * 10}\n\n{report.markdown_report}")
    print(f"\n{'=' * 10} Follow-up questions {'=' * 10}\n")
    for question in report.follow_up_questions:
        print(f"  - {question}")


if __name__ == "__main__":
    asyncio.run(main())
