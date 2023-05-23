from dataclasses import dataclass
from typing import List


@dataclass
class LaneAssignment:
    """Lane assignment class the holds the assigment of
        a trajectory to a lane segment.

    Args:
        lane_seg_id (int): Lane segment id.
        confidence (float): Assignment confidence.
        s_val (float): Frenet s-value of the matched point on the lane segment.
        d_val (float): Frenet d-value of the query location wrt the matched point
            on the lane segment.
        length (float): Centerline length of the assigned lane segment.
        matched_point (List[float, float]): x and y coordinates of the matched
            point on the lane segment.
    """

    lane_seg_id: int
    confidence: float
    s_val: float
    d_val: float
    length: float
    matched_point: List[float]
