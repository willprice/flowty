from flowty.cli import flow_method_base_parser
from flowty.cv.cuda_optflow import CudaTvL1OpticalFlow
from flowty.cv.optflow import TvL1OpticalFlow
from flowty.flow_command import AbstractFlowCommand


class TvL1FlowCommand(AbstractFlowCommand):
    def get_flow_algorithm(self, args):
        if args.cuda:
            if args.median_filtering:
                raise ValueError("Median filtering is not supported in CUDA TVL1 "
                                 "implementation")
            return CudaTvL1OpticalFlow(
                    tau=args.tau,
                    lambda_=getattr(args, 'lambda'),
                    theta=args.theta,
                    epsilon=args.epsilon,
                    gamma=args.gamma,
                    scale_count=args.scale_count,
                    warp_count=args.warp_count,
                    iterations=args.outer_iterations * args.inner_iterations,
                    scale_step=args.scale_step,
                    use_initial_flow=args.use_initial_flow,
            )
        else:
            median_filtering = args.median_filtering
            if median_filtering is None:
                median_filtering = 5
            return TvL1OpticalFlow(
                    tau=args.tau,
                    lambda_=getattr(args, 'lambda'),
                    theta=args.theta,
                    epsilon=args.epsilon,
                    gamma=args.gamma,
                    scale_count=args.scale_count,
                    warp_count=args.warp_count,
                    inner_iterations=args.inner_iterations,
                    outer_iterations=args.outer_iterations,
                    scale_step=args.scale_step,
                    median_filtering=median_filtering,
                    use_initial_flow=args.use_initial_flow,
            )

    @staticmethod
    def register_command(command_parsers):
        parser = command_parsers.add_parser("tvl1", parents=[flow_method_base_parser])
        parser.set_defaults(command=TvL1FlowCommand)
        parser.add_argument('--tau', type=float, default=0.25)
        parser.add_argument('--lambda', type=float, default=0.15)
        parser.add_argument('--theta', type=float, default=0.30)
        parser.add_argument('--gamma', type=float, default=0.0)
        parser.add_argument('--epsilon', type=float, default=0.01)
        parser.add_argument('--scale-count', type=int, default=3)
        parser.add_argument('--warp-count', type=int, default=5)
        parser.add_argument('--inner-iterations', type=int, default=30)
        parser.add_argument('--outer-iterations', type=int, default=10)
        parser.add_argument('--scale-step', type=float, default=0.8)
        parser.add_argument('--median-filtering', type=int, default=None)
        parser.add_argument('--use-initial-flow', action='store_true')
