import argparse

import flowty
from flowty.cli import flow_method_base_parser
from flowty.cv.optflow import VariationalRefinementOpticalFlow
from flowty.flow_command import AbstractFlowCommand


class VariationalRefinementCommand(AbstractFlowCommand):
    def get_flow_algorithm(self, args):
        return VariationalRefinementOpticalFlow(
            alpha=args.alpha,
            gamma=args.gamma,
            delta=args.delta,
            omega=args.omega,
            fixed_point_iterations=args.fixed_point_iterations,
            sor_iterations=args.sor_iterations,
        )

    @staticmethod
    def register_command(command_parsers):
        parser = command_parsers.add_parser(
            "vr",
            parents=[flow_method_base_parser],
            description="Compute Variational Refinement (Brox) optical flow",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.set_defaults(command=VariationalRefinementCommand)
        parser.add_argument("--alpha", type=float, default=20.0)
        parser.add_argument("--delta", type=float, default=5.0)
        parser.add_argument("--gamma", type=float, default=10.0)
        parser.add_argument("--omega", type=float, default=1.6)
        parser.add_argument("--fixed-point-iterations", type=int, default=5)
        parser.add_argument("--sor-iterations", type=int, default=5)
