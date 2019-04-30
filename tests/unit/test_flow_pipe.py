from flowty.flow_pipe import FlowPipe
import numpy as np


class RecordingDestination:
    def __init__(self):
        self.flow = []

    def write(self, flow):
        self.flow.append(flow)


class TestFlowPipe:
    def difference(reference, target):
        print("target: {}, reference: {}".format(target ,reference))
        return target - reference

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

    def test_input_transform_is_applied_to_reference_frames(self):
        flow = self.compute_flow([1, 2, 3],
                                 flow_algorithm=lambda reference, target: reference,
                                 input_transforms=[lambda f: f+1])
        assert flow == [np.array([2]), np.array([3])]

    def test_input_transform_is_applied_to_target_frames(self):
        flow = self.compute_flow([1, 2, 3],
                                 flow_algorithm=lambda reference, target: target,
                                 input_transforms=[lambda f: f+1])
        assert flow == [np.array([3]), np.array([4])]

    def test_output_transform_is_applied_to_flow(self):
        flow = self.compute_flow([1, 2, 3],
                                 output_transforms=[lambda f: f+1])
        assert flow == [np.array([2]), np.array([2])]

    def compute_flow(self, frames, flow_algorithm=difference, dilation=1, stride=1,
                     input_transforms=None, output_transforms=None):
        src = [np.array([f]) for f in frames]

        dest = RecordingDestination()
        pipe = FlowPipe(
                src, flow_algorithm, dest, dilation=dilation, stride=stride,
                input_transforms=input_transforms, output_transforms=output_transforms
        )
        pipe.run()
        return dest.flow



