from flow2 import Mat, VideoSource, VideoSink, DenseOpticalFlow, DualTVL1OpticalFlow, mat_to_np, np_to_mat, quantise_float_mat
import argparse
from pathlib import Path
from collections import deque

parser = argparse.ArgumentParser(
    description='Compute optical flow',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('src', type=Path)
parser.add_argument('dest', type=Path)
parser.add_argument('algorithm', choices=['brox', 'farneback', 'tvl1', 'dis', 'pca', 'variational-refinement'])
parser.add_argument('--video-stride', type=int, default=1)
parser.add_argument('--video-dilation', type=int, default=1)
parser.add_argument('--height', type=int,
                    help="Height of output flow frames."
                         "Defaults to input frame height if not provided")
parser.add_argument('--bound', default=20,
                    help="Max magnitude of flow, values above this are truncated.")
parser.add_argument('--cuda', action='store_true',
                    help="Use CUDA implementations where possible.")
parser.add_argument('--tvl1-tau', type=float, default=0.25)
parser.add_argument('--tvl1-lambda', type=float, default=0.15)
parser.add_argument('--tvl1-theta', type=float, default=0.30)
parser.add_argument('--tvl1-gamma', type=float, default=0.0)
parser.add_argument('--tvl1-epsilon', type=float, default=0.01)
parser.add_argument('--tvl1-scale-count', type=int, default=3)
parser.add_argument('--tvl1-warp-count', type=int, default=5)
parser.add_argument('--tvl1-inner-iterations', type=int, default=30)
parser.add_argument('--tvl1-outer-iterations', type=int, default=10)
parser.add_argument('--tvl1-scale-step', type=float, default=0.8)
parser.add_argument('--tvl1-median-filtering', type=int, default=5)
parser.add_argument('--tvl1-use-initial-flow', action='store_true')


class FlowPipe:
    def __init__(self, src, flow_algorithm, input_transforms, output_transforms, dest, stride=1, dilation=1):
        self.src = src
        self.flow_algorithm = flow_algorithm
        self.input_transforms = input_transforms
        self.output_transforms = output_transforms
        self.dest = dest
        self.stride = stride
        self.dilation = dilation

    def run(self):
        frame_queue = deque()
        frame_iter = iter(self.src)
        while len(frame_queue) < self.dilation:
            frame_queue.appendleft(next(frame_iter))
        assert len(frame_queue) == self.dilation

        for i, target in enumerate(frame_iter):
            reference = frame_queue.popleft()
            frame_queue.appendleft(target)
            assert len(frame_queue) == self.dilation - 1
            if i % self.stride == 0:
                flow = self.flow_algorithm(reference, target)
                self.write_flow(flow)

    def read_frame(self):
        frame = self.src.read_frame()
        for transform in self.input_transforms:
            frame = transform(frame)
        return frame

    def write_flow(self, flow):
        for transform in self.output_transforms:
            flow = transform(flow)
        self.dest.write_frame(flow)


def make_flow_algorithm(args) -> DenseOpticalFlow:
    if args.algorithm.lower() == 'tvl1':
        return DualTVL1OpticalFlow(
            tau=args.tvl1_tau,
            lambda_=args.tvl1_lambda,
            theta=args.tvl1_theta,
            epsilon=args.tvl1_epsilon,
            gamma=args.tvl1_gamma,
            scale_count=args.tvl1_scale_count,
            warp_count=args.tvl1_warps,
            inner_iterations=args.tvl1_inner_iterations,
            outer_iterations=args.tvl1_outer_iterations,
            scale_step=args.tvl1_scale_step,
            median_filtering=args.tvl1_median_filtering,
            use_initial_flow=args.use_initial_flow,
        )
    else:
        raise NotImplementedError(args.algorithm + " not implemented yet")


def flow_to_uv_pair(flow: Mat):
    array = mat_to_np(flow)
    array


def main(args):
    video_src = VideoSource(str(args.src.absolute()))
    video_sink = VideoSink(str(args.dest.absolute()))
    flow_algorithm = make_flow_algorithm(args)
    flow_quantisation = quantise_float_mat(bound=args.bound, min=0, max=255)
    pipeline = FlowPipe(
        video_src, [], flow_algorithm, [flow_to_uv_pair, flow_quantisation], video_sink,
        stride=args.video_stride,
        dilation=args.video_dilation
    )
    pipeline.run()


if __name__ == '__main__':
    main(parser.parse_args())