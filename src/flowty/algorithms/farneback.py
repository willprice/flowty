import argparse

from flowty.cli import flow_method_base_parser
from flowty.cv.cuda import get_cuda_enabled_device_count
from flowty.cv.cuda_optflow import CudaFarnebackOpticalFlow
from flowty.cv.optflow import FarnebackOpticalFlow
from flowty.flow_command import AbstractFlowCommand


class FarnebackCommand(AbstractFlowCommand):
    def get_flow_algorithm(self, args):
        if args.cuda:
            return CudaFarnebackOpticalFlow(
                    args.scale_count,
                    args.scale_factor,
                    args.fast_pyramids,
                    args.window_size,
                    args.iterations,
                    args.neighborhood_size,
                    args.poly_sigma
            )
        else:
            return FarnebackOpticalFlow(
                    args.scale_count,
                    args.scale_factor,
                    args.fast_pyramids,
                    args.window_size,
                    args.iterations,
                    args.neighborhood_size,
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
        parser.set_defaults(command=FarnebackCommand)
        parser.add_argument("--scale-count", type=int, default=5,
                            help="Number of levels in the image pyramid (including "
                                 "the original image). [1, infty)")
        parser.add_argument("--scale-factor", type=float, default=0.5,
                            help="Scale factor between each image pyramid level. "
                                 "(0, 1)")
        parser.add_argument("--fast-pyramids", action='store_true')
        parser.add_argument("--window-size", type=int, default=13,
                            help="Averaging window size at each pyramid level. "
                                 "Larger values increase noise robustness, but produce "
                                 "a more blurred flow field. [1, infty)")
        parser.add_argument("--iterations", type=int, default=10,
                            help="Number of iterations at each pyramid level.")
        parser.add_argument("--neighborhood-size", type=int, default=5,
                            help="Size of pixel neighborhood used to find polynomial "
                                 "expansion. [1, infty)")
        parser.add_argument("--poly-sigma", type=float, default=1.1,
                            help="Standard deviation of Gaussian used to smooth "
                                 "derivatives")
        parser.add_argument(
            "--cuda",
            action="store_true",
            help="Use CUDA implementation",
        )

    def main(self):
        if self.args.cuda:
            if get_cuda_enabled_device_count() < 1:
                raise RuntimeError("No CUDA devices available")
        super().main()
