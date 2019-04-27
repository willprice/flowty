from flowty.cv import mat_to_array
from flowty.cv.videoio import VideoSource
from flowty.flow_pipe import FlowPipe
from flowty.imgproc import quantise_flow
from flowty.methods import tvl1
from flowty.videoio import FlowImageWriter
from flowty.cv.cuda import get_cuda_enabled_device_count
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(
    description='Compute optical flow',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('src', type=Path)
parser.add_argument('dest', type=Path)
parser.add_argument('--video-stride', type=int, default=1)
parser.add_argument('--video-dilation', type=int, default=1)
parser.add_argument('--height', type=int,
                    help="Height of output flow frames."
                         "Defaults to input frame height if not provided")
parser.add_argument('--bound', default=20,
                    help="Max magnitude of flow, values above this are truncated.")
parser.add_argument('--cuda', action='store_true',
                    help="Use CUDA implementations where possible.")
command_parsers = parser.add_subparsers()


def add_subcommand_parser(command_parsers, command_spec):
    command_parser = command_parsers.add_parser(command_spec['command'])
    command_spec['parser_setup_fn'](command_parser)


add_subcommand_parser(command_parsers, tvl1.parser_spec)


def main(args=None):
    if args is None:
        args = parser.parse_args()
    if args.cuda:
        if get_cuda_enabled_device_count() < 1:
            raise RuntimeError("No CUDA devices available")
    video_src = VideoSource(str(args.src.absolute()))
    video_sink = FlowImageWriter(str(args.dest))
    flow_algorithm = args.flow_algorithm_getter(args)
    pipeline = FlowPipe(
        video_src, flow_algorithm, video_sink,
        output_transforms=[mat_to_array, quantise_flow],
        stride=args.video_stride,
        dilation=args.video_dilation
    )
    pipeline.run()


if __name__ == '__main__':
    main()
