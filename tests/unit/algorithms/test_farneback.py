import argparse

import pytest
from pytest import approx

from flowty.algorithms.farneback import FarnebackCommand
from flowty.cv.cuda_optflow import CudaFarnebackOpticalFlow
from flowty.cv.optflow import FarnebackOpticalFlow


class TestFarnebackCommand:
    arg_test_cases = [
        ("scale-count", "scale_count", 6),
        ("scale-factor", "scale_factor", 0.8),
        ("fast-pyramids", "use_fast_pyramids", (None, True)),
        ("window-size", "window_size", 20),
        ("neighborhood-size", "neighborhood_size", 10),
        ("poly-sigma", "poly_sigma", 1.5),
        ("iterations", "iterations", 20),
    ]

    @pytest.mark.parametrize("arg,attr,value", arg_test_cases)
    def test_cpu_args(self, arg, attr, value):
        if isinstance(value, tuple):
            cli_value, instance_value = value
        else:
            cli_value = value
            instance_value = value
        str_args = (
            ["farneback", "src", "dest", "--{}".format(arg)]
            + ([str(cli_value)] if cli_value is not None else [])
        )

        flow_alg = self.get_flow_alg(str_args)
        assert isinstance(flow_alg, FarnebackOpticalFlow)

        if isinstance(instance_value, float):
            expected_value = approx(instance_value)
        else:
            expected_value = instance_value
        assert getattr(flow_alg, attr) == expected_value

    @pytest.mark.skipif('not flowty.cuda_available')
    @pytest.mark.parametrize("arg,attr,value", arg_test_cases)
    def test_gpu_args(self, arg, attr, value):
        if isinstance(value, tuple):
            cli_value, instance_value = value
        else:
            cli_value = value
            instance_value = value
        str_args = (
                ["farneback", "src", "dest", "--{}".format(arg)]
                + ([str(cli_value)] if cli_value is not None else [])
                + ["--cuda"]
        )

        flow_alg = self.get_flow_alg(str_args)
        assert isinstance(flow_alg, CudaFarnebackOpticalFlow)

        if isinstance(instance_value, float):
            expected_value = approx(instance_value)
        else:
            expected_value = instance_value
        assert getattr(flow_alg, attr) == expected_value

    def get_flow_alg(self, str_args):
        parser = argparse.ArgumentParser()
        command_parsers = parser.add_subparsers()
        FarnebackCommand.register_command(command_parsers)
        args = parser.parse_args(str_args)
        command = FarnebackCommand(args)
        return command.get_flow_algorithm(args)
