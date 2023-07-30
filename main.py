openai_key = <my_key>

import textbase
from textbase.message import Message
from textbase import models
import os
from typing import List

# Load your OpenAI API key
models.OpenAI.api_key = openai_key
# or from environment variable:
# models.OpenAI.api_key = os.getenv("OPENAI_API_KEY")

# Prompt for GPT-3.5 Turbo
SYSTEM_PROMPT = """
You are a orderbot, an automated service to collect orders for a pizza restaurant. You first greet the 
customer, then collects the order, and then asks if it's a pickup or delivery. Then you summarize the 
order and check for a final time if the customer wants to add anything else. If it's a delivery, 
you ask for an address. Finally you give the bill and collect the payment. Make sure to clarify all 
options, extras and sizes to uniquely identify the item from the menu. You respond in a short, very 
conversational friendly style. 
The menu includes:
pepperoni pizza  12.95, 13.00, 14.00 
cheese pizza   10.95, 11.25, 12.50 
eggplant pizza   11.95, 12.05, 12.75 
fries 3.50, 4.50 
greek salad 7.25 
Toppings: 
extra cheese 2.00, 
mushrooms 1.50 
tomato sauce 1.50 
peppers 1.00 
Drinks: 
coke 1.00, 2.00, 3.00 
sprite 1.00, 2.00, 3.00 
bottled water 2.00 
"""


@textbase.chatbot("talking-bot")
def on_message(message_history: List[Message], state: dict = None):
    """Your chatbot logic here
    message_history: List of user messages
    state: A dictionary to store any stateful information

    Return a string with the bot_response or a tuple of (bot_response: str, new_state: dict)
    """

    if state is None or "counter" not in state:
        state = {"counter": 0}
    else:
        state["counter"] += 1

    # # Generate GPT-3.5 Turbo response
    bot_response = models.OpenAI.generate(
        system_prompt=SYSTEM_PROMPT,
        message_history=message_history,
        model="gpt-3.5-turbo",
        
    )

    return bot_response, state
