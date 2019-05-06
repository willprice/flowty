import argparse

import pytest
from pytest import approx

from flowty.algorithms.dis import DenseInverseSearchCommand
from flowty.cv.optflow import DenseInverseSearchOpticalFlow


class TestDenseInverseSearchCommand:
    @pytest.mark.parametrize(
        "arg,attr,value",
        [
            ("gradient-descent-iterations", "gradient_descent_iterations", 30),
            ("finest-scale", "finest_scale", 2),
            ("patch-size", "patch_size", 9),
            ("patch-stride", "patch_stride", 4),
            ("alpha", "alpha", 21.0),
            ("delta", "delta", 6.0),
            ("gamma", "gamma", 11.0),
            ("variational-refinement-iterations", "variational_refinement_iterations", 6),
            ("enable-spatial-propagation", "use_spatial_propagation", (None, True)),
            ("disable-spatial-propagation", "use_spatial_propagation", (None, False)),
            ("enable-mean-normalization", "use_mean_normalization", (None, True)),
            ("disable-mean-normalization", "use_mean_normalization", (None, False)),
        ],
    )
    def test_args(self, arg, attr, value):
        if isinstance(value, tuple):
            cli_value, instance_value = value
        else:
            cli_value = value
            instance_value = value
        str_args = ["dis", "src", "dest", "--preset", "medium", "--{}".format(arg)] + (
            [str(cli_value)] if cli_value is not None else []
        )

        flow_alg = self.get_flow_alg(str_args)

        assert isinstance(flow_alg, DenseInverseSearchOpticalFlow)
        if isinstance(instance_value, float):
            instance_value = approx(instance_value)
        assert getattr(flow_alg, attr) == instance_value

    def get_flow_alg(self, str_args):
        parser = argparse.ArgumentParser()
        command_parsers = parser.add_subparsers()
        DenseInverseSearchCommand.register_command(command_parsers)
        args = parser.parse_args(str_args)
        command = DenseInverseSearchCommand(args)
        return command.get_flow_algorithm(args)
