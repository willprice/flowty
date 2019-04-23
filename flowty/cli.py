from flowty.cv.core import Mat
from flowty.cv.videoio import VideoSource
from flowty.cv.optflow import TvL1OpticalFlow
from flowty.imgproc import quantise_flow
from flowty.videoio import FlowImageWriter
from flowty.cv.cuda import get_cuda_enabled_device_count
from flowty.cv.cuda_optflow import CudaTvL1OpticalFlow
import argparse
from pathlib import Path
from collections import deque
import time

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
    def __init__(self,
                 src,
                 flow_algorithm,
                 dest,
                 input_transforms=None,
                 output_transforms=None,
                 stride=1, dilation=1):
        self.src = src
        self.flow_algorithm = flow_algorithm
        self.input_transforms = input_transforms if input_transforms is not None else []
        self.output_transforms = output_transforms if output_transforms is not None else []
        self.dest = dest
        self.stride = stride
        self.dilation = dilation

    def run(self):
        frame_queue = deque()
        frame_iter = iter(self.src)
        while len(frame_queue) < self.dilation:
            frame_queue.appendleft(next(frame_iter))
        assert len(frame_queue) == self.dilation

        t = time.time()
        for i, target in enumerate(frame_iter):
            data_load_time = time.time() - t
            print("Data time (ms): ", data_load_time * 1e3)
            reference = frame_queue.popleft()
            frame_queue.appendleft(target)
            assert len(frame_queue) == self.dilation
            if i % self.stride == 0:
                t = time.time()
                flow = self.flow_algorithm(reference, target)
                flow_time = (time.time() - t) * 1e3
                print("Flow time (ms): ", flow_time)

                t = time.time()
                self.write_flow(flow)
                write_time = (time.time() - t) * 1e3
                print("Write time (ms): ", write_time)
            t = time.time()

    def write_flow(self, flow):
        for transform in self.output_transforms:
            flow = transform(flow)
        self.dest.write(flow)


def make_flow_algorithm(args):
    if args.algorithm.lower() == 'tvl1':
        if args.cuda:
            return CudaTvL1OpticalFlow(
                tau=args.tvl1_tau,
                lambda_=args.tvl1_lambda,
                theta=args.tvl1_theta,
                epsilon=args.tvl1_epsilon,
                gamma=args.tvl1_gamma,
                scale_count=args.tvl1_scale_count,
                warp_count=args.tvl1_warp_count,
                iterations=args.tvl1_outer_iterations * args.tvl1_inner_iterations,
                scale_step=args.tvl1_scale_step,
                use_initial_flow=args.tvl1_use_initial_flow,
            )
        else:
            return TvL1OpticalFlow(
                tau=args.tvl1_tau,
                lambda_=args.tvl1_lambda,
                theta=args.tvl1_theta,
                epsilon=args.tvl1_epsilon,
                gamma=args.tvl1_gamma,
                scale_count=args.tvl1_scale_count,
                warp_count=args.tvl1_warp_count,
                inner_iterations=args.tvl1_inner_iterations,
                outer_iterations=args.tvl1_outer_iterations,
                scale_step=args.tvl1_scale_step,
                median_filtering=args.tvl1_median_filtering,
                use_initial_flow=args.tvl1_use_initial_flow,
            )
    else:
        raise NotImplementedError(args.algorithm + " not implemented yet")


def mat_to_array(mat):
    return mat.asarray()


def array_to_mat(array):
    return Mat.fromarray(array)


def main(args=None):
    if args is None:
        args = parser.parse_args()
    if args.cuda:
        if get_cuda_enabled_device_count() < 1:
            raise RuntimeError("No CUDA devices available")
    video_src = VideoSource(str(args.src.absolute()))
    video_sink = FlowImageWriter(str(args.dest))
    flow_algorithm = make_flow_algorithm(args)
    pipeline = FlowPipe(
        video_src, flow_algorithm, video_sink,
        output_transforms=[mat_to_array, quantise_flow],
        stride=args.video_stride,
        dilation=args.video_dilation
    )
    pipeline.run()


if __name__ == '__main__':
    main()
