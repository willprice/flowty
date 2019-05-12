import argparse

import pytest
from pytest import approx

from flowty.algorithms.vr import VariationalRefinementCommand
from flowty.cv.optflow import VariationalRefinementOpticalFlow


class TestVariationalRefinmentCommand:
    @pytest.mark.parametrize(
        "arg,attr,value",
        [
            ("alpha", "alpha", 21.0),
            ("gamma", "gamma", 11.0),
            ("delta", "delta", 4.0),
            ("omega", "omega", 1.2),
            ("fixed-point-iterations", "fixed_point_iterations", 4),
            ("sor-iterations", "sor_iterations", 4),
        ],
    )
    def test_args(self, arg, attr, value):
        str_args = (
            ["vr", "src", "flow/{axis}/frame_{index:05d}.jpg", "--{}".format(arg)]
            + ([str(value)] if value is not None else [])
        )

        flow_alg = self.get_flow_alg(str_args)
        assert isinstance(flow_alg, VariationalRefinementOpticalFlow)

        if isinstance(value, float):
            value = approx(value)

        assert getattr(flow_alg, attr) == value

    def get_flow_alg(self, str_args):
        parser = argparse.ArgumentParser()
        command_parsers = parser.add_subparsers()
        VariationalRefinementCommand.register_command(command_parsers)
        args = parser.parse_args(str_args)
        command = VariationalRefinementCommand(args)
        return command.get_flow_algorithm(args)
