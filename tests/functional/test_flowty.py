from pathlib import Path
import numpy as np
from imageio import imread
from numpy.testing import assert_array_equal

from flowty import flowty
from ..resources import GOLD_DIR, RUBBER_WHALE


class TestFlowty:
    def test_tvl1_flow_from_mp4_to_uv_images(self, tmpdir):
        src = RUBBER_WHALE["media_path"]["png"]
        output_dir = tmpdir / "flow"
        dest = output_dir / "{axis}" / "frame{index:02d}.png"
        bound = 20
        dilation = 1
        stride = 2

        flowty.main(
            [
                "dis",
                "--video-stride",
                str(stride),
                "--video-dilation",
                str(dilation),
                "--bound",
                str(bound),
                str(src),
                str(dest),
            ]
        )
        gold_dir = (
            GOLD_DIR
            / "dis"
            / "stride={stride}-dilation={dilation}-bound={bound}".format(
                stride=stride, dilation=dilation, bound=bound
            )
            / "rubber-whale"
        )
        for axis in ["u", "v"]:
            for i in range(1, RUBBER_WHALE["frame_count"] // stride):
                flow_frame_path = Path(output_dir / axis / "frame{:02d}.png".format(i))
                gold_frame_path = Path(gold_dir / axis / "frame{:02d}.png".format(i))
                assert flow_frame_path.exists()
                flow_frame = imread(flow_frame_path)
                gold_frame = imread(gold_frame_path)
                assert flow_frame.shape == RUBBER_WHALE["resolution"]
                assert flow_frame.dtype == np.uint8
                assert_array_equal(flow_frame, gold_frame)
            assert not Path(
                output_dir
                / axis
                / "frame{:02d}.png".format(RUBBER_WHALE["frame_count"])
            ).exists()
