"""Code for timing various things, to test performance.

To create a new timeable action, either subclass Timeable and implement the run method,
or use the `register_timeable` decorator to register a callable as a timeable.

To time a timeable, use the `time_timeable` function. To time all timeables, use the
`time_all_timeables` function.

Examples
--------
Create a new timeable and time it:
>>> from pvg.timing import Timeable, register_timeable, time_timeable
>>> @register_timeable(name="my_timeable")
>>> class MyTimeable(Timeable):
...     def __init__(self):
...         pass
...     def run(self, profiler):
...         pass
>>> time_timeable("my_timeable")
"""

from .timeables import Timeable, register_timeable, time_timeable, time_all_timeables
from .models import ModelTimeable
