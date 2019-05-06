import argparse

import pytest
from pytest import approx

import flowty
from flowty.algorithms.brox import BroxCommand
from flowty.cv.cuda_optflow import CudaBroxOpticalFlow


@pytest.mark.skipif('not flowty.cuda_available')
class TestBroxFlowCommand:
    @pytest.mark.parametrize("arg,attr,value", [
        ("alpha", "alpha", 0.2),
        ("gamma", "gamma", 50.0),
        ("inner-iterations", "inner_iterations", 10),
        ("outer-iterations", "outer_iterations", 100),
        ("solver-iterations", "solver_iterations", 100),
        ("scale-factor", "scale_factor", 0.5),
    ])
    def test_cpu_args(self, arg, attr, value):
        str_args = ["brox", "src", "dest", "--{}".format(arg), str(value)]

        flow_alg = self.get_flow_alg(str_args)

        assert isinstance(flow_alg, CudaBroxOpticalFlow)
        if isinstance(value, float):
            value = approx(value)
        assert getattr(flow_alg, attr) == value

    def get_flow_alg(self, str_args):
        parser = argparse.ArgumentParser()
        command_parsers = parser.add_subparsers()
        BroxCommand.register_command(command_parsers)
        args = parser.parse_args(str_args)
        command = BroxCommand(args)
        flow_alg = command.get_flow_algorithm(args)
        return flow_alg

    def test_raises_runtime_error_if_cuda_not_available(self, monkeypatch):
        with monkeypatch.context() as ctx:
            ctx.setattr(flowty, "cuda_available", False)
            with pytest.raises(RuntimeError):
                self.get_flow_alg(["brox", "src", "dest"])

