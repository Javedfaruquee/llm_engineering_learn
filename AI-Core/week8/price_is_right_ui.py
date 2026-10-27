"""Week 8, Day 5: The Price is Right finale - a Gradio user interface for the deal-hunting agents.

The UI is built up piece by piece; choose how far to go with the stage argument:

    python day5_price_is_right_ui.py header    # just a title and description
    python day5_price_is_right_ui.py table     # plus a table of deals found so far (example data)
    python day5_price_is_right_ui.py alerts    # plus: click a row to get a push alert for that deal
    python day5_price_is_right_ui.py reset     # trim memory.json back to the first 2 deals
    python day5_price_is_right_ui.py           # the full app (price_is_right.py): agents run
                                               # on load and every 5 minutes, with live logs

Each UI stage opens in your browser and runs until you stop it with Ctrl+C.
The full app needs everything from days 1-4: the Modal pricer-service, the
products_vectorstore, deep_neural_network.pth, and the OpenAI / Anthropic / Pushover keys.
"""

import argparse

import gradio as gr

from agents.deals import Deal, Opportunity
from deal_agent_framework import DealAgentFramework


TITLE = "The Price is Right"
TABLE_HEADERS = ["Description", "Price", "Estimate", "Discount", "URL"]


def example_opportunities():
    """A single made-up opportunity, so the table has something to show before any agent runs."""
    deal = Deal(product_description="Example description", price=100.0, url="https://cnn.com")
    return [Opportunity(deal=deal, estimate=200.0, discount=100.0)]


def table_for(opportunities):
    """Rows for the deals table: description, price, estimate, discount, URL."""
    return [[opp.deal.product_description, opp.deal.price, opp.estimate, opp.discount, opp.deal.url]
            for opp in opportunities]


def build_ui(stage):
    """Build one stage of the UI.

    Args:
        stage: "header", "table" or "alerts" (see the module docstring).

    Returns:
        The gr.Blocks app, ready to launch.
    """
    # Only the alerts stage needs the agents, and creating them is slow
    agent_framework = None
    if stage == "alerts":
        agent_framework = DealAgentFramework()
        agent_framework.init_agents_as_needed()

    with gr.Blocks(title=TITLE, fill_width=True) as ui:
        if stage == "header":
            with gr.Row():
                gr.Markdown(f'<div style="text-align: center;font-size:24px">{TITLE} - Deal Hunting Agentic AI</div>')
            with gr.Row():
                gr.Markdown('<div style="text-align: center;font-size:14px">Autonomous agent framework that finds online deals, '
                            'collaborating with a proprietary fine-tuned LLM deployed on Modal, and a RAG pipeline with a '
                            'frontier model and Chroma.</div>')
            return ui

        # Per-session state holding the opportunities shown in the table
        opportunities = gr.State(example_opportunities())

        with gr.Row():
            gr.Markdown(f'<div style="text-align: center;font-size:24px">"{TITLE}" - Deal Hunting Agentic AI</div>')
        with gr.Row():
            gr.Markdown('<div style="text-align: center;font-size:14px">Deals surfaced so far:</div>')
        with gr.Row():
            # max_height (not height) since Gradio 5
            opportunities_dataframe = gr.Dataframe(
                headers=TABLE_HEADERS,
                wrap=True,
                column_widths=[4, 1, 1, 1, 2],
                row_count=10,
                col_count=5,
                max_height=400,
            )

        # Fill the table when the page loads
        ui.load(table_for, inputs=[opportunities], outputs=[opportunities_dataframe])

        if agent_framework:
            def do_select(opportunities, selected: gr.SelectData):
                """Send a push alert for the opportunity in the clicked row."""
                agent_framework.planner.messenger.alert(opportunities[selected.index[0]])

            opportunities_dataframe.select(do_select, inputs=[opportunities], outputs=[])

    return ui


def main():
    """Launch the chosen UI stage, reset the memory, or run the full app."""
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument("stage", nargs="?", default="app", choices=["header", "table", "alerts", "reset", "app"])
    stage = parser.parse_args().stage

    if stage == "reset":
        DealAgentFramework.reset_memory()
        print("memory.json now holds the first 2 deals only")
    elif stage == "app":
        from price_is_right import App
        App().run()
    else:
        build_ui(stage).launch(inbrowser=True)


if __name__ == "__main__":
    main()
