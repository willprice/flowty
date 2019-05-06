import argparse

import flowty
from flowty.cli import flow_method_base_parser
from flowty.cv.cuda_optflow import CudaPyramidalLucasKanade
from flowty.flow_command import AbstractFlowCommand


class PyrLucasKanadeCommand(AbstractFlowCommand):
    def get_flow_algorithm(self, args):
        if not flowty.cuda_available:
            raise RuntimeError("CUDA-accelerated device not available. Pyramidal "
                               "Lucas-Kanade is not implemented on the CPU.")
        return CudaPyramidalLucasKanade(
            window_size=args.window_size,
            max_scales=args.max_scales,
            iterations=args.iterations,
        )

    @staticmethod
    def register_command(command_parsers):
        parser = command_parsers.add_parser(
            "pyrlk",
            parents=[flow_method_base_parser],
            description="Compute Pyramidal Lucas-Kanade optical flow",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.set_defaults(command=PyrLucasKanadeCommand)
        parser.add_argument(
                "--window-size",
                type=int,
                default=13,
                help="Size of window for over-constraining linear equation",
        )
        parser.add_argument(
            "--iterations",
            type=int,
            default=30,
            help="Number of iterations",
        )
        parser.add_argument(
                "--max-scales",
                type=int,
                default=3,
                help="Maximum number of scales in the image pyramid",
        )
