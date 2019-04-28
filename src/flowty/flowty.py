import argparse
from flowty.algorithms import tvl1
import sys

parser = argparse.ArgumentParser(
        description="Compute optical flow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
command_parsers = parser.add_subparsers()
tvl1.TvL1FlowCommand.register_command(command_parsers)


def main(args=None):
    if args is None:
        args = parser.parse_args()

    if not hasattr(args, 'command'):
        parser.print_help()
        sys.exit(1)

    command = args.command(args)
    command.main()


if __name__ == "__main__":
    main()
