import json
import requests
from openai import OpenAI
from pvg.constants import OPENAI_API_KEY, OPENROUTER_API_KEY


def get_openai_response(
    model,
    messages,
    api_key=OPENAI_API_KEY,
    temperature=1.0,
    log_probs=False,
    top_logprobs=None,
    num_responses=1,
):

    client = OpenAI(api_key=api_key)

    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        logprobs=log_probs,
        top_logprobs=top_logprobs,
        n=num_responses,
    ).choices


def get_openrouter_response(
    model,
    messages,
    api_key=OPENROUTER_API_KEY,
    temperature=1.0,
    log_probs=False,
    top_logprobs=None,
    num_responses=1,
    force_multiple_generations=False,
):
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
    responses = []

    # Crazily, the openrouter API doesn't support multiple completions in a single request, so we have to make multiple requests
    if "openai" not in model or force_multiple_generations:
        completions = [
            requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                data=json.dumps(
                    {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "logprobs": log_probs,
                        "top_logprobs": top_logprobs,
                    }
                ),
            ).json()["choices"][0]
            for _ in range(num_responses)
        ]

    else:
        completions = get_openai_response(
            model.split("/")[-1],
            messages,
            temperature=temperature,
            log_probs=log_probs,
            top_logprobs=top_logprobs,
            num_responses=num_responses,
        )
        completions = [c.dict() for c in completions]

    for completion in completions:

        responses.append(
            {
                "message": completion["message"]["content"],
                "log_probs": (
                    [
                        {k: c[k] for k in ["token", "logprob"]}
                        for c in completion["logprobs"]["content"]
                    ]
                    if log_probs and completion["logprobs"] is not None
                    else None
                ),
                "top_logprobs": (
                    [
                        [
                            {k: cc[k] for k in ["token", "logprob"]}
                            for cc in c["top_logprobs"]
                        ]
                        for c in completion["logprobs"]["content"]
                    ]
                    if top_logprobs and completion["logprobs"] is not None
                    else None
                ),
            }
        )

    return responses
