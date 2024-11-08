"""Features of Python 3.11 for use with lower versions of Python.

In the Dockerfile we use Python 3.10, so we can't use the new features of Python 3.11.

These are copied from the python source code.
"""

from .enum import StrEnum
from .typing import TypedDict, NotRequired
