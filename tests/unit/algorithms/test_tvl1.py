import argparse

import pytest
from pytest import approx

import flowty
from flowty.algorithms.tvl1 import TvL1Command
from flowty.cv.cuda_optflow import CudaTvL1OpticalFlow
from flowty.cv.optflow import TvL1OpticalFlow


class TestTvL1FlowCommand:
    @pytest.mark.parametrize("arg,attr,value", [
        ("tau", "tau", 0.2),
        ("lambda", "lambda_", 0.1),
        ("theta", "theta", 0.4),
        ("gamma", "gamma", 0.1),
        ("epsilon", "epsilon", 0.02),
        ("scale-count", "scale_count", 3),
        ("warp-count", "warp_count", 3),
        ("inner-iterations", "inner_iterations", 20),
        ("outer-iterations", "outer_iterations", 20),
        ("scale-step", "scale_step", 0.7),
        ("median-filtering", "median_filtering", 2),
    ])
    def test_cpu_args(self, arg, attr, value):
        str_args = ["tvl1", "src", "flow/{axis}/frame_{index:05d}.jpg", "--{}".format(arg), str(value)]

        flow_alg = self.get_flow_alg(str_args)

        assert isinstance(flow_alg, TvL1OpticalFlow)
        if isinstance(value, float):
            value = approx(value)
        assert getattr(flow_alg, attr) == value

    def get_flow_alg(self, str_args):
        parser = argparse.ArgumentParser()
        command_parsers = parser.add_subparsers()
        TvL1Command.register_command(command_parsers)
        args = parser.parse_args(str_args)
        command = TvL1Command(args)
        flow_alg = command.get_flow_algorithm(args)
        return flow_alg

    @pytest.mark.skipif('not flowty.cuda_available')
    @pytest.mark.parametrize("arg,attr,value", [
        ("tau", "tau", 0.2),
        ("lambda", "lambda_", 0.1),
        ("theta", "theta", 0.4),
        ("gamma", "gamma", 0.1),
        ("epsilon", "epsilon", 0.02),
        ("scale-count", "scale_count", 3),
        ("warp-count", "warp_count", 3),
        ("scale-step", "scale_step", 0.7),
    ])
    def test_gpu_args(self, arg, attr, value):
        str_args = ["tvl1", "src", "flow/{axis}/frame_{index:05d}.jpg", "--{}".format(arg), str(value), "--cuda"]

        flow_alg = self.get_flow_alg(str_args)

        assert isinstance(flow_alg, CudaTvL1OpticalFlow)
        if isinstance(value, float):
            value = approx(value)
        assert getattr(flow_alg, attr) == value

    @pytest.mark.skipif('not flowty.cuda_available')
    def test_gpu_iterations(self):
        inner_iterations = 20
        outer_iterations = 30
        str_args = ["tvl1", "src", "flow/{axis}/frame_{index:05d}.jpg", "--inner-iterations", str(inner_iterations),
                    "--outer-iterations", str(outer_iterations), "--cuda"]

        flow_alg = self.get_flow_alg(str_args)

        assert isinstance(flow_alg, CudaTvL1OpticalFlow)
        assert flow_alg.iterations == inner_iterations * outer_iterations

    @pytest.mark.skipif('not flowty.cuda_available')
    def test_median_filtering_on_gpu_raises_error(self):
        str_args = ["tvl1", "src", "flow/{axis}/frame_{index:05d}.jpg", "--median-filtering", "3", "--cuda"]

        with pytest.raises(ValueError):
            self.get_flow_alg(str_args)

    def test_raises_runtime_error_if_cuda_not_available(self, monkeypatch):
        with monkeypatch.context() as ctx:
            ctx.setattr(flowty, "cuda_available", False)
            with pytest.raises(RuntimeError):
                self.get_flow_alg(["tvl1", "src", "flow/{axis}/frame_{index:05d}.jpg", "--cuda"])