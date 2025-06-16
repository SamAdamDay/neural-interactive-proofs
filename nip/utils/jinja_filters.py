"""Custom filters for Jinja2 templates."""


def capitalise_first_letter(value: str) -> str:
    """Capitalise the first letter of a string.

    Parameters
    ----------
    value : str
        The string to capitalise.

    Returns
    -------
    capitalised_value : str
        The string with the first letter capitalised, and the rest unchanged.
    """
    if not value:
        return value
    return value[0].upper() + value[1:]


def add_s_plural(value: str, count: int) -> str:
    """Add 's' to a string if the count is not 1.

    Parameters
    ----------
    value : str
        The string to pluralise.
    count : int
        The count of the items.

    Returns
    -------
    pluralised_value : str
        The pluralised string.
    """
    if count == 1:
        return value
    return value + "s"
