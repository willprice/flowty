import argparse

import pytest

from flowty.algorithms.pyrlk import PyrLucasKanadeCommand
from flowty.cv.cuda_optflow import CudaPyramidalLucasKanade


@pytest.mark.skipif('not flowty.cuda_available')
class TestPyramidalLucasKanadeCommand:
    @pytest.mark.parametrize(
        "arg,attr,value",
        [
            ("window-size", "window_size", 14),
            ("iterations", "iterations", 40),
            ("max-scales", "max_scales", 4),
        ],
    )
    def test_args(self, arg, attr, value):
        str_args = (
            ["pyrlk", "src", "dest", "--{}".format(arg)]
            + ([str(value)] if value is not None else [])
        )

        flow_alg = self.get_flow_alg(str_args)
        assert isinstance(flow_alg, CudaPyramidalLucasKanade)

        assert getattr(flow_alg, attr) == value

    def get_flow_alg(self, str_args):
        parser = argparse.ArgumentParser()
        command_parsers = parser.add_subparsers()
        PyrLucasKanadeCommand.register_command(command_parsers)
        args = parser.parse_args(str_args)
        command = PyrLucasKanadeCommand(args)
        return command.get_flow_algorithm(args)
