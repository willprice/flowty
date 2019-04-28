import argparse

from flowty.cv.cuda import get_cuda_enabled_device_count
from flowty.algorithms import tvl1

parser = argparse.ArgumentParser(
        description="Compute optical flow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
        "--cuda", action="store_true", help="Use CUDA implementations where possible."
)
command_parsers = parser.add_subparsers()
tvl1.TvL1FlowCommand.register_command(command_parsers)


def main(args=None):
    if args is None:
        args = parser.parse_args()

    if args.cuda:
        if get_cuda_enabled_device_count() < 1:
            raise RuntimeError("No CUDA devices available")

    command = args.command(args)
    command.main()


if __name__ == "__main__":
    main()
