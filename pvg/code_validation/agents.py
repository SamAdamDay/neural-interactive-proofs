import json
import requests

# from pvg.constants import OPENROUTER_API_KEY

# For some reason above the import above doesn't work
OPENROUTER_API_KEY = (
    "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"
)


def get_response(model, messages):
    """
    Sends a POST request to the OpenRouter API to get responses from a chat model.

    Parameters:
    - model (str): The name of the chat model to use.
    - messages (list): A list of dictionaries representing the chat messages. Each dictionary should have a "role" key with the value "user" or "assistant", and a "content" key with the content of the message.

    Returns:
    - response (Response): The response object returned by the API.

    Raises:
    - requests.exceptions.RequestException: If there was an error sending the request.

    """

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        data=json.dumps({"model": model, "messages": messages}),
    )

    return response.json()["choices"][0]["message"]["content"]
