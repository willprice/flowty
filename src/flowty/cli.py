import argparse
from pathlib import Path

flow_method_base_parser = argparse.ArgumentParser(
    add_help=False,  # child parsers will automatically add help.
)
flow_method_base_parser.add_argument(
    "src", type=Path, help="Path to video source, " "e.g. /data/video.mp4"
)
flow_method_base_parser.add_argument(
    "dest",
    type=Path,
    help="Path to video output, e.g. /data/flow/{axis}/frame_{:06d}.jpg",
)
flow_method_base_parser.add_argument(
    "--opencv-videoio-backend",
    default="ffmpeg",
    choices=["ffmpeg", "gstreamer"],
    help="OpenCV VideoCapture backend",
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
