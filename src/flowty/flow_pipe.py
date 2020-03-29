import time
from collections import deque
from typing import Iterator

from tqdm import tqdm

from flowty.cv import Mat


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

    def _frame_generator(self) -> Iterator:
        for frame in iter(self.src):
            for transform in self.input_transforms:
                frame = transform(frame)
            yield frame

    def run(self):
        frame_iter = self._frame_generator()
        frame_queue = deque()
        while len(frame_queue) < self.dilation:
            frame_queue.append(next(frame_iter))
        assert len(frame_queue) == self.dilation

        t = time.time()
        try:
            total = int(self.src.frame_count)
        except AttributeError:
            total = None

        pbar = tqdm(enumerate(frame_iter), total=total, dynamic_ncols=True)
        for i, target in pbar:
            data_load_time = (time.time() - t) * 1e3
            reference = frame_queue.popleft()
            frame_queue.append(target)
            assert len(frame_queue) == self.dilation
            if i % self.stride == 0:
                t = time.time()
                flow = self.flow_algorithm(reference, target)
                flow_time = (time.time() - t) * 1e3

                t = time.time()
                self.write_flow(flow)
                write_time = (time.time() - t) * 1e3
                pbar.set_description("read: {:.2f}ms, compute: {:.2f}ms, write: {:.2f}ms".format(
                        data_load_time, flow_time, write_time
                ))
            t = time.time()

    def write_flow(self, flow: Mat) -> None:
        for transform in self.output_transforms:
            flow = transform(flow)
        self.dest.write(flow)