import time
from collections import deque


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