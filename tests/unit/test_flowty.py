import sys
from io import StringIO

from unittest.mock import Mock

from flowty.flowty import main, command_parsers


class IOCapture:
    def __init__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._stdout = StringIO()
        self._stderr = StringIO()

    def __enter__(self):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

    @property
    def stdout(self):
        return self._stdout.getvalue()

    @property
    def stderr(self):
        return self._stderr.getvalue()


class TestFlowtyCli:
    def test_prints_help_when_no_command_is_provided(self):
        with IOCapture() as capture:
            main([])
            assert 'usage: flowty' in capture.stdout

    def test_invokes_registered_command(self):
        tvl1_parser = command_parsers.add_parser('tvl1')
        tvl1_command = Mock()
        tvl1_parser.set_defaults(command=tvl1_command)

        main(['tvl1'])

        tvl1_command.assert_called_once()