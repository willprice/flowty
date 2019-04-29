from flowty.flow_pipe import FlowPipe
import numpy as np


class RecordingDestination:
    def __init__(self):
        self.flow = []

    def write(self, flow):
        self.flow.append(flow)


class TestFlowPipe:
    def test_1_flow_field_between_2_frames(self):
        flow = self.compute_flow([1, 2])
        assert len(flow) == 1
        assert flow[0] == np.array([1])

    def test_2_flow_fields_between_3_frames(self):
        flow = self.compute_flow([1, 2, 4])
        assert len(flow) == 2
        assert flow == [np.array([1]), np.array([2])]

    def test_1_flow_field_between_3_frames_when_dilation_is_2(self):
        flow = self.compute_flow([1, 2, 3], dilation=2)
        assert len(flow) == 1
        assert flow[0] == np.array([2])

    def test_2_flow_fields_between_4_frames_when_dilation_is_2(self):
        flow = self.compute_flow([1, 2, 3, 5], dilation=2)
        assert flow == [np.array([2]), np.array([3])]

    def compute_flow(self, frames, dilation=1, stride=1):
        src = [np.array([f]) for f in frames]

        def flow_algorithm(reference, target):
            print("target: {}, reference: {}".format(target ,reference))
            return target - reference

        dest = RecordingDestination()
        pipe = FlowPipe(
                src, flow_algorithm, dest, dilation=dilation, stride=stride
        )
        pipe.run()
        return dest.flow
