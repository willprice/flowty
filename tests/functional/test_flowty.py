from pathlib import Path
import numpy as np
from imageio import imread

from flowty import flowty
from tests.resources import RUBBER_WHALE


class TestFlowty:
    def test_tvl1_flow_from_mp4_to_uv_images(self, tmpdir):
        src = RUBBER_WHALE['media_path']['png']
        output_dir = tmpdir / 'flow'
        dest = output_dir / '{axis}' / 'frame{index:02d}.png'
        stride = 2

        flowty.main(["dis", "--video-stride", str(stride), str(src), str(dest)])

        for axis in ['u', 'v']:
            for i in range(1, RUBBER_WHALE['frame_count'] // stride):
                flow_frame_path = Path(output_dir / axis / 'frame{:02d}.png'.format(i))
                assert flow_frame_path.exists()
                flow_frame = imread(flow_frame_path)
                assert flow_frame.shape == RUBBER_WHALE['resolution']
                assert flow_frame.dtype == np.uint8
            assert not Path(output_dir / axis / 'frame{:02d}.png'.format( RUBBER_WHALE['frame_count'])).exists()
