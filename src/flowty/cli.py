import argparse
from typing import Callable
from pathlib import Path

flow_method_base_parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    add_help=False  # child parsers will automatically add help.
)
flow_method_base_parser.add_argument("src", type=Path, help="Path to video source")
flow_method_base_parser.add_argument("dest", type=Path)
flow_method_base_parser.add_argument(
    "--opencv-videoio-backend", default="ffmpeg", help="OpenCV VideoCapture backend"
)
flow_method_base_parser.add_argument(
    "--video-stride",
    type=int,
    default=1,
    help="Number of frames between consecutive reference frames. "
    "e.g. set to 2 to skip every other frame.",
)
flow_method_base_parser.add_argument(
    "--video-dilation",
    type=int,
    default=1,
    help="Distance between reference and target in number of frames. "
    "e.g. set to 2 to to compute flow between frames n and n + 2.",
)
flow_method_base_parser.add_argument(
    "--bound", default=20, help="Max magnitude of flow, values above this are clipped."
)


