"""Script to time the timeable action"""

from argparse import ArgumentParser

from pvg.timing import time_timeable, time_all_timeables

parser = ArgumentParser(description="Time a timeable action")
parser.add_argument(
    "-a", "--all", action="store_true", help="Time all available timeables"
)
parser.add_argument(
    "names",
    metavar="NAME",
    type=str,
    nargs="*",
    help="The names of the timeables to time",
)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.all:
        time_all_timeables(print_results=True)
    elif not args.names:
        parser.print_usage()
    else:
        for name in args.names:
            time_timeable(name, print_results=True)
