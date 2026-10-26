import gradio as gr
from dotenv import load_dotenv

from pro_implementation.answer import DB_NAME, answer_question, collection

load_dotenv(override=True)


def as_text(content):
    """Gradio 6 stores message content as a list of parts like [{"type": "text", "text": "..."}]."""
    if isinstance(content, str):
        return content
    return " ".join(part.get("text", "") for part in content if isinstance(part, dict))


def short_source(source):
    """Show 'company/about.md' rather than the full path the ingest stored."""
    source = source.replace("\\", "/")
    if "knowledge-base/" in source:
        return source.split("knowledge-base/", 1)[1]
    return source


def format_context(context):
    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"
    for doc in context:
        result += f"<span style='color: #ff7800;'>Source: {short_source(doc.metadata['source'])}</span>\n\n"
        result += doc.page_content + "\n\n"
    return result


def chat(history):
    if not history or history[-1]["role"] != "user":
        return history, gr.skip()      # nothing new was asked (e.g. Enter on an empty box)
    last_message = as_text(history[-1]["content"])
    prior = [{"role": m["role"], "content": as_text(m["content"])} for m in history[:-1]]
    try:
        answer, context = answer_question(last_message, prior)
    except Exception as error:
        history.append({"role": "assistant", "content": f"⚠️ Sorry, something went wrong: {error}"})
        return history, gr.skip()
    history.append({"role": "assistant", "content": answer})
    return history, format_context(context)


def main():
    if collection.count() == 0:
        print(f"The vector database in {DB_NAME} is empty. Build it first with:\n"
              "    python week5/pro_implementation/ingest.py")
        return

    def put_message_in_chatbot(message, history):
        if not message.strip():
            return "", history
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="Insurellm Expert Assistant") as ui:
        gr.Markdown("# 🏢 Insurellm Expert Assistant\nAsk me anything about Insurellm!")

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(label="💬 Conversation", height=600, buttons=["copy"])
                message = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about Insurellm...",
                    show_label=False,
                )

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    label="📚 Retrieved Context",
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=600,
                )

        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(chat, inputs=chatbot, outputs=[chatbot, context_markdown])

    ui.launch(inbrowser=True, theme=theme)  # Gradio 6 takes the theme in launch()


if __name__ == "__main__":
    main()