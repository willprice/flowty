import argparse

from flowty.cli import flow_method_base_parser
from flowty.cv.cuda_optflow import CudaTvL1OpticalFlow
from flowty.cv.optflow import TvL1OpticalFlow
from flowty.flow_command import AbstractFlowCommand


class TvL1FlowCommand(AbstractFlowCommand):
    def get_flow_algorithm(self, args):
        if args.cuda:
            if args.median_filtering:
                raise ValueError(
                    "Median filtering is not supported in CUDA TVL1 " "implementation"
                )
            return CudaTvL1OpticalFlow(
                tau=args.tau,
                lambda_=getattr(args, "lambda"),
                theta=args.theta,
                epsilon=args.epsilon,
                gamma=args.gamma,
                scale_count=args.scale_count,
                warp_count=args.warp_count,
                iterations=args.outer_iterations * args.inner_iterations,
                scale_step=args.scale_step,
                use_initial_flow=False,
            )
        else:
            median_filtering = args.median_filtering
            if median_filtering is None:
                median_filtering = 5
            return TvL1OpticalFlow(
                tau=args.tau,
                lambda_=getattr(args, "lambda"),
                theta=args.theta,
                epsilon=args.epsilon,
                gamma=args.gamma,
                scale_count=args.scale_count,
                warp_count=args.warp_count,
                inner_iterations=args.inner_iterations,
                outer_iterations=args.outer_iterations,
                scale_step=args.scale_step,
                median_filtering=median_filtering,
                use_initial_flow=False,
            )

    @staticmethod
    def register_command(command_parsers):
        parser = command_parsers.add_parser(
            "tvl1",
            parents=[flow_method_base_parser],
            description="Compute TV-L1 optical flow",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.set_defaults(command=TvL1FlowCommand)
        parser.add_argument(
            "--tau", type=float, default=0.25, help="Time step of the numerical scheme."
        )
        parser.add_argument(
            "--lambda",
            type=float,
            default=0.15,
            help="Weight parameter for the data term, attachment parameter. "
            "This determines the smoothness of the output. "
            "The smaller this parameter is, the smoother the solutions we obtain. "
            "It depends on the range of motions of the images, so its value should"
            " be adapted to each image sequence",
        )
        parser.add_argument(
            "--theta",
            type=float,
            default=0.30,
            help="Weight parameter for (u - v)^2, tightness parameter. "
            "It serves as a link between the attachment and the regularization terms."
            "In theory, it should have a small value in order to maintain both parts in"
            " correspondence. "
            "The method is stable for a large range of values of this parameter.",
        )
        parser.add_argument(
            "--gamma",
            type=float,
            default=0.0,
            help="Coefficient for additional illumination variation term",
        )
        parser.add_argument(
            "--epsilon",
            type=float,
            default=0.01,
            help="Stopping criterion threshold used in the numerical scheme, which is a"
            " trade-off between precision and running time.  "
            "A small value will yield more accurate solutions at the expense of a "
            "slower convergence.",
        )
        parser.add_argument(
            "--scale-count",
            type=int,
            default=5,
            help="Number of scales used to create the pyramid of images.",
        )
        parser.add_argument(
            "--warp-count",
            type=int,
            default=5,
            help="Number of warpings per scale. "
            "Represents the number of times that I1(x+u0) and grad(I1(x+u0)) "
            "are computed per scale. "
            "This is a parameter that assures the stability of the method. "
            "It also affects the running time, so it is a compromise between "
            "speed and accuracy.",
        )
        parser.add_argument(
            "--inner-iterations",
            type=int,
            default=30,
            help="Inner iterations (between outlier filtering) used in the numerical "
            "scheme.",
        )
        parser.add_argument(
            "--outer-iterations",
            type=int,
            default=10,
            help="Outer iterations (number of inner loops) used in the numerical "
            "scheme.",
        )
        parser.add_argument(
            "--scale-step", type=float, default=0.8, help="Step between scales (<1)"
        )
        parser.add_argument(
            "--median-filtering",
            type=int,
            default=None,
            help="Median filter kernel size (1 = no filter) (3 or 5). CPU only "
            "(default: 3 for CPU, disabled for GPU)",
        )
        parser.add_argument(
            "--cuda",
            action="store_true",
            help="Use CUDA implementations where possible.",
        )
