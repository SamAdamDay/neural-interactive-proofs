"""Script to time the timeable action."""

from argparse import ArgumentParser
from textwrap import indent

from nip.timing import time_timeable, time_all_timeables, list_timeables

parser = ArgumentParser(description="Time a timeable action")
parser.add_argument(
    "-a", "--all", action="store_true", help="Time all available timeables"
)
parser.add_argument(
    "-l", "--list", action="store_true", help="List all available timeables"
)
parser.add_argument(
    "names",
    metavar="NAME",
    type=str,
    nargs="*",
    help="The names of the timeables to time",
)
parser.add_argument(
    "--param-scale",
    type=float,
    help="The scale factor for key default parameters",
    default=1.0,
)
parser.add_argument(
    "--no-record-shapes",
    help="Don't record tensor shapes",
    action="store_false",
    dest="record_shapes",
    default=True,
)
parser.add_argument(
    "--no-profile-memory",
    help="Don't profile memory usage",
    action="store_false",
    dest="profile_memory",
    default=True,
)
parser.add_argument(
    "--with-stack",
    help="Record the stack trace. This introduces a significant additional overhead",
    action="store_true",
    dest="with_stack",
)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.list:
        print("Available timeables:")  # noqa: T201
        for name, timeable in list_timeables():
            print(f"  {name}:")  # noqa: T201
            print(indent(timeable.__doc__.strip(), "    "))  # noqa: T201
    elif args.all:
        time_all_timeables(print_results=True)
    elif not args.names:
        parser.print_usage()
    else:
        for name in args.names:
            time_timeable(
                name,
                print_results=True,
                param_scale=args.param_scale,
                record_shapes=args.record_shapes,
                profile_memory=args.profile_memory,
                with_stack=args.with_stack,
            )
