import argparse
import string
from pathlib import Path
import numpy as np

from flowty.imgproc import quantise_flow
from .cv.imgcodecs import imwrite


def parse_template_fields(str_template):
    return {field for (_, field, _, _) in string.Formatter().parse(
            str_template)}


def get_flow_writer(args: argparse.Namespace):
    extension_writer_map = [
        (['jpeg', 'jpg', 'png'], FlowUVImageWriter),
        (['np', 'npy'], FlowNumpyWriter)
    ]
    for extensions, writer_cls in extension_writer_map:
        for extension in extensions:
            if args.dest.lower().endswith('.' + extension):
                return writer_cls(args.dest)
    else:
        raise ValueError("Unable to retrieve flow writer for '{}'".format(
                args.dest))


class FlowUVImageWriter:
    def __init__(self, file_path_template: str, bound=20):
        template_fields = parse_template_fields(file_path_template)
        if 'axis' not in template_fields:
            raise ValueError("Missing '{axis}' substitution in output template")
        if 'index' not in template_fields:
            raise ValueError("Missing '{index}' substitution in output template")
        self.file_path_template = file_path_template
        self.frame_index = 1
        self.bound = bound

    def write(self, flow: np.ndarray) -> None:
        if flow.ndim != 3:
            raise ValueError("Expected flow to be 3D, but was {}D".format(flow.ndim))
        if flow.shape[2] != 2:
            raise ValueError("Expected flow to have 2 channels, but had {}".format(flow.shape[2]))

        quantised_flow = quantise_flow(flow, bound=self.bound)
        self._write_uv_images(quantised_flow)
        self.frame_index += 1

    def _write_uv_images(self, flow: np.ndarray) -> None:
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


class FlowNumpyWriter:
    def __init__(self, file_path_template: str):
        if 'index' not in parse_template_fields(file_path_template):
            raise ValueError("Missing '{index}' substitution in output template")
        self.file_path_template = file_path_template
        self.frame_index = 1

    def write(self, flow: np.ndarray) -> None:
        if flow.ndim != 3:
            raise ValueError("Expected flow to be 3D, but was {}D".format(flow.ndim))
        if flow.shape[2] != 2:
            raise ValueError("Expected flow to have 2 channels, but had {}".format(flow.shape[2]))

        self._write_flow(flow)

        self.frame_index += 1

    def _write_flow(self, flow: np.ndarray) -> None:
        filepath = Path(self.file_path_template.format(index=self.frame_index))
        filepath.parent.mkdir(exist_ok=True, parents=True)
        with filepath.open(mode='wb') as f:
            np.save(f, flow)

