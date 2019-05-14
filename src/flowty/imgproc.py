import numpy as np


def quantise_flow(flow: np.ndarray, bound: float = 20) -> np.ndarray:
    """Quantise 2D vector field
    Args:
        flow: 2D float-valued vector field.
        bound: Max value of flow (and -bound is min value), values above this
            are clipped.
    """
    flow = np.clip(flow, -bound, bound)
    flow_range = 2 * bound
    flow += bound
    flow *= (255 / flow_range)
    quantised_flow = flow.astype(np.uint8)
    return quantised_flow


def flow_to_hsv(flow: np.ndarray, bound: float = 20) -> np.ndarray:
    """Create HSV flow representation """
    raise NotImplementedError()
