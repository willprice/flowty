import argparse

import flowty
from flowty.cli import flow_method_base_parser
from flowty.cv.cuda_optflow import CudaBroxOpticalFlow
from flowty.flow_command import AbstractFlowCommand


class BroxCommand(AbstractFlowCommand):
    def get_flow_algorithm(self, args):
        if not flowty.cuda_available:
            raise RuntimeError("CUDA-accelerated device not available. Brox is not "
                               "implemented on the CPU.")
        return CudaBroxOpticalFlow(
            alpha=args.alpha,
            gamma=args.gamma,
            scale_factor=args.scale_factor,
            inner_iterations=args.inner_iterations,
            outer_iterations=args.outer_iterations,
            solver_iterations=args.solver_iterations,
        )

    @staticmethod
    def register_command(command_parsers):
        parser = command_parsers.add_parser(
            "brox",
            parents=[flow_method_base_parser],
            description="Compute Brox optical flow",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.set_defaults(command=BroxCommand)
        parser.add_argument(
                "--alpha",
                type=float,
                default=0.197,
                help="Flow smoothness factor. (0, 1)",
        )
        parser.add_argument(
            "--gamma",
            type=float,
            default=50.0,
            help="Weight of gradient constancy term. (0, infty)",
        )
        parser.add_argument(
            "--inner-iterations",
            type=int,
            default=5,
            help="Number of inner (lagged non-linearity) iterations."
                 ,
        )
        parser.add_argument(
            "--outer-iterations",
            type=int,
            default=150,
            help="Number of outer (warping) iterations.",
        )
        parser.add_argument(
            "--solver-iterations",
            type=int,
            default=10,
            help="Number of inner (linear system solver) iterations."
        )
        parser.add_argument(
            "--scale-factor", type=float, default=0.8,
                help="Step between image pyramid scales. (0, 1)"
        )
