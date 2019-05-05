import argparse

import flowty
from flowty.cli import flow_method_base_parser
from flowty.cv.optflow import DenseInverseSearchOpticalFlow
from flowty.flow_command import AbstractFlowCommand


class DenseInverseSearchCommand(AbstractFlowCommand):
    def get_flow_algorithm(self, args):
        return DenseInverseSearchOpticalFlow(preset=args.preset)

    @staticmethod
    def register_command(command_parsers):
        parser = command_parsers.add_parser(
            "dis",
            parents=[flow_method_base_parser],
            description="Compute Dense Inverse Search optical flow",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.set_defaults(command=DenseInverseSearchCommand)
        parser.add_argument(
            "--preset",
            type=str,
            choices=["ultrafast", "fast", "medium"],
            default="fast",
        )
