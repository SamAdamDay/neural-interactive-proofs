"""Utilities for input and output."""

from typing import Optional, Literal


def yes_no_user_prompt(
    query_message: str,
    initial_message: Optional[str] = None,
    default_answer: Optional[Literal["y", "n"]] = None,
) -> Literal["y", "n"]:
    """Prompt the user with a yes or no questions.

    This function will keep prompting the user until a valid response is provided.

    Parameters
    ----------
    query_message : str
        The message to ask the user each time.
    initial_message : str
        An initial message to print before the query loop.
    default_answer: Literal["y", "n"], optional
        The default answer, which will be used if the user just presses <return>.

    Returns
    -------
    option_selected : Literal["y", "n"]
        The option selected by the user.
    """

    if initial_message is not None:
        print(initial_message)  # noqa: T201

    if default_answer == "y":
        yn_prompt = "[Y/n]"
    elif default_answer == "n":
        yn_prompt = "[y/N]"
    else:
        yn_prompt = "[y/n]"

    while True:
        response = input(f"{query_message} {yn_prompt}: ")
        if response.lower() == "y":
            return "y"
        elif response.lower() == "n":
            return "n"
        elif response == "" and default_answer is not None:
            return default_answer
        else:
            print("Invalid response. Please enter 'y' or 'n'.")  # noqa: T201
