import argparse
from flowty.algorithms import tvl1, brox, pyrlk
import sys

parser = argparse.ArgumentParser(
        prog='flowty',
        description="Compute optical flow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
command_parsers = parser.add_subparsers()
tvl1.TvL1FlowCommand.register_command(command_parsers)
brox.BroxFlowCommand.register_command(command_parsers)
pyrlk.PyrLKFlowCommand.register_command(command_parsers)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parser.parse_args(argv)

    if not hasattr(args, 'command'):
        parser.print_help()
        return 1

    command = args.command(args)
    command.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
