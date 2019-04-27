from pathlib import Path
import numpy as np
from .cv.imgcodecs import imwrite


class FlowImageWriter:
    def __init__(self, file_path_template: str):
        self.file_path_template = file_path_template
        self.frame_index = 1

    def __call__(self, flow: np.ndarray):
        if flow.ndim != 3:
            raise ValueError("Expected flow to be 3D, but was {}D".format(flow.ndim))
        if flow.shape[2] != 2:
            raise ValueError("Expected flow to have 2 channels, but had {}".format(flow.shape[2]))
        if flow.dtype != np.uint8:
            raise ValueError("Expected flow to be np.uint8, but was {}".format(flow.dtype))
        u_img_path = self.file_path_template.format(
            axis='u',
            index=self.frame_index
        )
        v_img_path = self.file_path_template.format(
            axis='v',
            index=self.frame_index
        )
        for dest in [u_img_path, v_img_path]:
            Path(dest).parent.mkdir(exist_ok=True, parents=True)
        imwrite(u_img_path, flow[..., 0])
        imwrite(v_img_path, flow[..., 1])

        self.frame_index += 1

    def write(self, flow: np.ndarray):
        return self(flow)
