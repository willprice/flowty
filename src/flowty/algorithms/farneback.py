import argparse

from flowty.cli import flow_method_base_parser
from flowty.cv.cuda import get_cuda_enabled_device_count
from flowty.cv.cuda_optflow import CudaFarnebackOpticalFlow
from flowty.cv.optflow import FarnebackOpticalFlow
from flowty.flow_command import AbstractFlowCommand


class FarnebackFlowCommand(AbstractFlowCommand):
    def get_flow_algorithm(self, args):
        if args.cuda:
            return CudaFarnebackOpticalFlow(
                    args.scale_count,
                    args.scale_factor,
                    args.fast_pyramids,
                    args.window_size,
                    args.iterations,
                    args.poly_count,
                    args.poly_sigma
            )
        else:
            return FarnebackOpticalFlow(
                    args.scale_count,
                    args.scale_factor,
                    args.fast_pyramids,
                    args.window_size,
                    args.iterations,
                    args.poly_count,
                    args.poly_sigma
            )

    @staticmethod
    def register_command(command_parsers):
        parser = command_parsers.add_parser(
            "farneback",
            parents=[flow_method_base_parser],
            description="Compute Farneback optical flow",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.set_defaults(command=FarnebackFlowCommand)
        parser.add_argument("--scale-count", type=int, default=5,
                            help="Number of scales in pyramid")
        parser.add_argument("--scale-factor", type=float, default=0.5,
                            help="Scale between each image pyramid")
        parser.add_argument("--fast-pyramids", action='store_true')
        parser.add_argument("--window-size", type=int, default=13)
        parser.add_argument("--iterations", type=int, default=10)
        parser.add_argument("--poly-count", type=int, default=5)
        parser.add_argument("--poly-sigma", type=float, default=1.1)
        parser.add_argument(
            "--cuda",
            action="store_true",
            help="Use CUDA implementations where possible.",
        )

    def main(self):
        if self.args.cuda:
            if get_cuda_enabled_device_count() < 1:
                raise RuntimeError("No CUDA devices available")
        super().main()
