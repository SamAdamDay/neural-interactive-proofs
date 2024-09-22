import json
import requests


def get_openrouter_response(model, messages, api_key):
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
        headers={"Authorization": f"Bearer {api_key}"},
        data=json.dumps({"model": model, "messages": messages}),
    )

    return response.json()["choices"][0]["message"]["content"]
