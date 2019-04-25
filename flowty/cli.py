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
from queue import Queue
from threading import Thread
import threading

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


class FlowWorkItem:
    def __init__(self, target, reference, index, time):
        self.target = target
        self.reference = reference
        self.index = index
        self.time = time

class WriterWorkItem:
    def __init__(self, flow, index):
        self.flow = flow
        self.index = index


class FlowPipe:
    def __init__(self,
                 src,
                 flow_algorithm,
                 dest,
                 output_transforms=None,
                 stride=1, dilation=1):
        self.src = src
        self.flow_algorithm = flow_algorithm
        self.output_transforms = output_transforms if output_transforms is not None else []
        self.dest = dest
        self.stride = stride
        self.dilation = dilation
        self.flow_work_queue = Queue(maxsize=20)
        self.writer_work_queue = Queue(maxsize=20)
        self.n_threads = 4
        self.n_frames_processed_lock = threading.Lock()
        self.n_frames_processed = 0

    def run(self):
        frame_reading_worker_thread = Thread(target=self.frame_reading_worker)
        flow_worker_threads = [Thread(target=self.flow_compute_worker)
                               for _ in range(self.n_threads)]
        flow_writing_worker_thread = Thread(target=self.flow_writing_worker)
        threads = flow_worker_threads + [frame_reading_worker_thread, flow_writing_worker_thread]

        start = time.time()
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()
        duration_ms = (time.time() - start) * 1e3
        avg_flow_time = duration_ms / self.n_frames_processed
        print("avg flow time (ms): ", avg_flow_time)

    def frame_reading_worker(self):
        frame_queue = deque()
        frame_iter = iter(self.src)
        while len(frame_queue) < self.dilation:
            frame_queue.appendleft(next(frame_iter))
        assert len(frame_queue) == self.dilation

        for i, target in enumerate(frame_iter):
            reference = frame_queue.popleft()
            frame_queue.appendleft(target)
            assert len(frame_queue) == self.dilation
            if i % self.stride == 0:
                print("read frame")
                self.flow_work_queue.put(FlowWorkItem(target=target,
                                                      reference=reference,
                                                      index=i, time=time.time()))
        for i in range(self.n_threads):
            self.flow_work_queue.put(None)

    def flow_compute_worker(self):
        alg = self.flow_algorithm()
        while True:
            work_item = self.flow_work_queue.get()
            if work_item is None:
                break
            reference = work_item.reference
            target = work_item.target
            flow = alg(reference, target)
            with self.n_frames_processed_lock:
                self.n_frames_processed += 1
            print("compute flow")
            print("flow time (ms): ", (time.time() - work_item.time) * 1e3)
            self.writer_work_queue.put(WriterWorkItem(flow=flow, index=work_item.index))
            self.flow_work_queue.task_done()
        self.writer_work_queue.put(None)

    def flow_writing_worker(self):
        while True:
            work_item = self.writer_work_queue.get()
            if work_item is None:
                break
            flow = work_item.flow
            for transform in self.output_transforms:
                flow = transform(flow)
            self.dest.write(flow, index=work_item.index)
            print("write flow")
            self.writer_work_queue.task_done()


def make_flow_algorithm(args):
    if args.algorithm.lower() == 'tvl1':
        if args.cuda:
            return lambda: CudaTvL1OpticalFlow(
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
            return lambda: TvL1OpticalFlow(
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
