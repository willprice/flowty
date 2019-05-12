from abc import ABC

from flowty.cv import mat_to_array
from flowty.cv.videoio import VideoSource
from flowty.flow_pipe import FlowPipe
from flowty.videoio import get_flow_writer


class AbstractFlowCommand(ABC):

    def __init__(self, args):
        self.args = args
        self.video_src = VideoSource(
                str(args.src.absolute()), backend=args.opencv_videoio_backend
        )
        self.video_sink = get_flow_writer(args)
        self.flow_algorithm = self.get_flow_algorithm(args)

    def get_flow_algorithm(self, args):
        raise NotImplementedError()

    @staticmethod
    def register_command(command_parsers):
        raise NotImplementedError()

    def main(self):

        pipeline = FlowPipe(
                src=self.video_src,
                flow_algorithm=self.flow_algorithm,
                dest=self.video_sink,
                output_transforms=[mat_to_array],
                stride=self.args.video_stride,
                dilation=self.args.video_dilation,
        )
        pipeline.run()