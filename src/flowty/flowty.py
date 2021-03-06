import argparse
from flowty.algorithms import tvl1, brox, pyrlk, farneback, vr, dis
import sys

parser = argparse.ArgumentParser(
        prog='flowty',
        description="Compute optical flow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
command_parsers = parser.add_subparsers()

tvl1.TvL1Command.register_command(command_parsers)
brox.BroxCommand.register_command(command_parsers)
pyrlk.PyrLucasKanadeCommand.register_command(command_parsers)
farneback.FarnebackCommand.register_command(command_parsers)
vr.VariationalRefinementCommand.register_command(command_parsers)
dis.DenseInverseSearchCommand.register_command(command_parsers)


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
