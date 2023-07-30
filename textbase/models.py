import openai

from textbase.message import Message
import json
import pandas as pd
import numpy as np

# Creating function description for price comparision.
function_descriptions = [
    {
        "name": "other_restaurant_price_info",
        "description": "Get price of item from different restaurant",
        "parameters": {
            "type": "object",
            "properties": {
                "restaurant_name": {
                    "type": "string",
                    "description": "The name of the other restaurant, e.g. pizzahut",
                },
                "ordered_item": {
                    "type": "string",
                    "description": "The ordered item, e.g. cheese pizza",
                },
                "ordered_item_size": {
                    "type": "string",
                    "description": "The ordered item size, e.g. small",
                },
            },
            "required": ["restaurant_name", "ordered_item", "ordered_item_size"],
        },
    }
]

# Created Datasets of known restaurants menu with price.
pizzahut_data= pd.DataFrame([['large',13.15,11.55,12.95,5.50,4.00,4.00], ['medium',11.00,10.15,7.80,np.nan,3.00,3.00], 
                  ['small',8.00,7.20,7.45,4.50,2.00,2.00]],columns= ['size','pepperoni pizza', 'cheese pizza','eggplant pizza','fries','coke','sprite'])
dominos_data= pd.DataFrame([['large',13.85,11.95,13.15,5.80,4.50,4.50], ['medium',11.50,10.85,8.00,np.nan,3.50,3.50], 
                  ['small',8.50,7.50,7.85,4.90,2.50,2.50]],columns= ['size','pepperoni pizza', 'cheese pizza','eggplant pizza','fries','coke','sprite'])


# Defining the function for calling
def other_restaurant_price_info(restaurant_name, ordered_item, ordered_item_size):
    """Get price information of items from other restaurants."""

    if 'pizzahut' in restaurant_name.lower():
        price_info = {
            "restaurant_name": restaurant_name,
            "ordered_item": ordered_item,
            "ordered_item_size": ordered_item_size,
            "price": pizzahut_data[pizzahut_data['size']== ordered_item_size][ordered_item].item()
        }
    elif 'domino' in restaurant_name.lower():
        price_info = {
            "restaurant_name": restaurant_name,
            "ordered_item": ordered_item,
            "ordered_item_size": ordered_item_size,
            "price": dominos_data[dominos_data['size']== ordered_item_size][ordered_item].item()
        }
    else:
        price_info = {
            "restaurant_name": restaurant_name,
            "ordered_item": ordered_item,
            "ordered_item_size": ordered_item_size,
            "price": "No Data Available"
        }

    return json.dumps(price_info)

class OpenAI:
    api_key = None

    @classmethod
    def generate(
        cls,
        system_prompt: str,
        message_history: list[Message],
        model="gpt-3.5-turbo",
        max_tokens=3000,
        temperature=0.5,
    ):
        assert cls.api_key is not None, "OpenAI API key is not set"
        openai.api_key = cls.api_key

        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                *map(dict, message_history),
            ],
            functions=function_descriptions,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response1= response["choices"][0]["message"]

        # Checking if the function is called for generating the output.
        if "function_call" in response1:

            params = json.loads(response1.function_call.arguments)

            chosen_function = eval(response1.function_call.name)
            price = chosen_function(**params)

            response2 = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *map(dict, message_history),
                    {"role": "function", "name": response1.function_call.name, "content": price},
                ],
                functions=function_descriptions,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response2["choices"][0]["message"]["content"]
            
        # Returning generated output without calling any function.
        else:
            return response["choices"][0]["message"]["content"]
