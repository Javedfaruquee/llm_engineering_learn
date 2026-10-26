import os
from dotenv import load_dotenv
from openai import OpenAI

# Read the API keys from the .env file
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Create one client per company.
# All three accept the same kind of request, so we can use the same OpenAI class,
# we just point it at a different web address (base_url) for Anthropic and Google.
openai_client = OpenAI(api_key=openai_api_key)
anthropic_client = OpenAI(api_key=anthropic_api_key, base_url="https://api.anthropic.com/v1/")
gemini_client = OpenAI(api_key=google_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

# Which model each bot uses
gpt_model = "gpt-4.1-mini"
claude_model = "claude-haiku-4-5"
gemini_model = "gemini-3.5-flash-lite"   # check this name is available on your account

# Every bot gets this text added to its instructions so it knows it is in a group chat
group_chat_rules = (
    " You are in a group chat with two other chatbots. "
    "You will be shown the chat so far, with each line starting with the speaker's name. "
    "Reply as yourself only. Do not write your own name at the start of your reply, "
    "and do not write replies for the other bots."
    "Ensure to end the conversation with a logical conclusion."
)

# The personality of each bot
gpt_system = """You are GPT, a chatbot who is very argumentative; you disagree with anything 
in the conversation and you challenge everything, in a snarky way. Keep your responses short
and witty, between 1 and 2 sentences.""" + group_chat_rules

claude_system = """You are Claude, a very polite, courteous chatbot. You try to agree with
 everything the others say, or find common ground. If someone is argumentative, you try to
 calm them down and keep chatting. Keep your responses short and polite, between 1 and 2 
 sentences.""" + group_chat_rules

gemini_system = """You are Gemini, a chatbot who is an arbitrator; you try to mediate 
disputes between the others and find balanced solutions. Keep your responses short and 
informative, between 1 and 2 sentences.""" + group_chat_rules


# ---------------------------------------------------------------
# The whole conversation is stored in ONE list.
# Each item is a pair: (who spoke, what they said).
# ---------------------------------------------------------------
transcript = [
    ("Host", "Welcome to the chat, GPT, Claude and Gemini! Say hello to each other."),
]


# ---------------------------------------------------------------
# Build the messages to send to ONE bot.
#
# We turn the whole transcript into a block of text, one line per speaker,
# put it in a single user message, and tell the bot it is its turn.
# ---------------------------------------------------------------
def build_messages(bot_name, system_prompt):
    # Turn the transcript into one block of text
    chat_so_far = ""
    for speaker, text in transcript:
        chat_so_far = chat_so_far + speaker + ": " + text + "\n"

    # Ask the bot for its next line
    user_message = (
        "Here is the group chat so far:\n\n"
        + chat_so_far
        + "\nIt is now your turn, " + bot_name + ". Write your next reply."
    )

    # print ("Build message for #### " + bot_name + "\n"
    #        "\n" + "System prompt #### " + system_prompt + "\n"
    #        "\n" + "User message #### " + user_message + "\n")

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


# ---------------------------------------------------------------
# One function per bot: build its messages, send them, return the reply
# ---------------------------------------------------------------
def call_gpt():
    messages = build_messages("GPT", gpt_system)
    response = openai_client.chat.completions.create(model=gpt_model, messages=messages)
    return response.choices[0].message.content

def call_claude():
    messages = build_messages("Claude", claude_system)
    response = anthropic_client.chat.completions.create(model=claude_model, messages=messages)
    return response.choices[0].message.content

def call_gemini():
    messages = build_messages("Gemini", gemini_system)
    response = gemini_client.chat.completions.create(model=gemini_model, messages=messages)
    return response.choices[0].message.content

# Save a new line in the transcript and print it
def record(speaker, text):
    transcript.append((speaker, text))
    print("### " + speaker + ":")
    print(text)
    print()

# ---------------------------------------------------------------
# Run the conversation for at least 5 turns
# ---------------------------------------------------------------
print("### Host:")
print(transcript[0][1])
print()

for i in range(5):
    gpt_reply = call_gpt()
    record("GPT", gpt_reply)

    claude_reply = call_claude()
    record("Claude", claude_reply)

    gemini_reply = call_gemini()
    record("Gemini", gemini_reply)