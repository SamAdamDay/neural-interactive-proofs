"""Utilities for timing pieces of code."""

from datetime import datetime


class TimeInfoPrinter:
    """Class to print the current time and a message.

    Parameters
    ----------
    num_digits : int, default=8
        The number of digits to print in the time difference
    """

    FORCE_DISABLE = False

    def __init__(self, num_digits: int = 8):
        self.num_digits = num_digits
        self._enabled = True
        self.previous_time = None

    @property
    def enabled(self):
        """Whether the printer is enabled."""
        return self._enabled and not self.FORCE_DISABLE

    def enable(self):
        """Enable the printer."""
        self._enabled = True

    def disable(self):
        """Disable the printer."""
        self._enabled = False

    def set_enabled_state(self, state: bool):
        """Set the enabled state.

        Parameters
        ----------
        state : bool
            The state to set
        """
        self._enabled = state

    def print(self, message: str = ""):
        """Print the current time and the message.

        Parameters
        ----------
        message : str, default=""
            The message to print
        """
        current_time = datetime.now()
        if self.enabled:
            if self.previous_time is not None:
                time_diff = (current_time - self.previous_time).total_seconds()
                time_diff = round(time_diff, self.num_digits)
                print(  # noqa: T201
                    f"{current_time.strftime('%H:%M:%S.%f')} "
                    f"[{time_diff:{self.num_digits+1}.{self.num_digits}f}] "
                    f"{message}"
                )
            else:
                print(f"{current_time.strftime('%H:%M:%S.%f')} {message}")  # noqa: T201
        self.previous_time = current_time
