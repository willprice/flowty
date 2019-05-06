import argparse

from flowty.cli import flow_method_base_parser
from flowty.cv.optflow import DenseInverseSearchOpticalFlow
from flowty.flow_command import AbstractFlowCommand


class DenseInverseSearchCommand(AbstractFlowCommand):
    def get_flow_algorithm(self, args):
        return DenseInverseSearchOpticalFlow(
            preset=args.preset,
            gradient_descent_iterations=args.gradient_descent_iterations,
            finest_scale=args.finest_scale,
            patch_size=args.patch_size,
            patch_stride=args.patch_stride,
            use_mean_normalization=args.mean_normalization,
            use_spatial_propagation=args.spatial_propagation,
            alpha=args.alpha,
            delta=args.delta,
            gamma=args.gamma,
            variational_refinement_iterations=args.variational_refinement_iterations,
        )

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
        parser.add_argument(
            "--gradient-descent-iterations",
            type=int,
            help="Number of gradient descent iterations in the patch "
            "inverse stage. Higher values may improve quality in some cases.",
        )
        parser.add_argument(
            "--finest-scale",
            type=int,
            help="Number of levels in the image pyramid. [0, infty)",
        )
        parser.add_argument(
            "--patch-size",
            type=int,
            help="Size of an image patch for matching. 8x8 patches "
            "work well enough in most cases.",
        )
        parser.add_argument(
            "--patch-stride",
            type=int,
            help="Stride between neighbour patches. Lower values "
            "correspond to higher quality. (patch-size, infty)",
        )
        parser.add_argument(
            "--enable-mean-normalization",
            dest="mean_normalization",
            action="store_true",
            help="Enable mean normalization of patches when computing patch "
            "distance. Improves robustness to changes in illumination.",
        )
        parser.add_argument(
            "--disable-mean-normalization",
            dest="mean_normalization",
            action="store_false",
            help="Disable mean normalization of matches when computing patch "
            "distance. Only disable if you don't have any illumination "
            "changes.",
        )
        parser.add_argument(
            "--enable-spatial-propagation",
            dest="spatial_propagation",
            action="store_true",
            help="Enable spatial propagation of good flow vectors. Can help "
            "recover from errors introduced by coarse-to-fine scheme.",
        )
        parser.add_argument(
            "--disable-spatial-propagation",
            dest="spatial_propagation",
            action="store_false",
            help="Disable spatial propagation of good flow vectors. Disabling can "
            "make flow a bit smoother.",
        )
        parser.add_argument(
            "--alpha", type=float, help="Weight of the smoothness term. (0, infty)"
        )
        parser.add_argument(
            "--delta", type=float, help="Weight of the color constancy term. (0, infty)"
        )
        parser.add_argument(
            "--gamma",
            type=float,
            help="Weight of the gradient " "constancy term. (0, infty)",
        )
        parser.add_argument(
            "--variational-refinement-iterations",
            type=int,
            help="Number of fixed point iterations of variational "
            "refinement per scale. Set to 0 to disable "
            "variational refinement completely. Higher values "
            "will produce smoother and higher quality flow.",
        )
