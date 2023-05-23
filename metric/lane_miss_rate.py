import os
from operator import attrgetter
from pathlib import Path
from typing import Dict, List, Tuple

import av2.geometry.interpolate as interp_utils
import numpy as np

from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap
from numpy.typing import NDArray
from rtree import index
from shapely.geometry import LineString, Point
from tqdm.contrib.concurrent import process_map

from metric.lane_assignment import LaneAssignment
from viz.scenario_visualization import visualize_scenario

FIXED_PLOT_SIZE = (120, 120)

DATASET_SAMPLING_FREQUENCY = 10  # Hertz

VELOCITY_TRHESHOLD_FACTOR = 0.2  # Meter / (Meter / Second) = Second
VELOCITY_TRHESHOLD_OFFSET = 0.7  # Meter

DISTANCE_THRESHOLD = 5.0  # Meter
ORIENTATION_ESTIMATION_STEP_SIZE = 0.001  # Meter
ORIENTATION_THRESHOLD = np.pi  # Rad
ORIENTATION_WEIGHTING = 0.5  # -

ASSIGNMENT_CONFIDENCE_THRESHOLD = 0.1  # -


class LaneMissRate:
    def __init__(
        self,
        dataset_root: str,
        plot_dir: str = None,
    ):
        """Lane Miss Rate (LMR) metric class.

        Args:
            dataset_root (str): Dataset root path.
            plot_dir (str, optional): Output directory of scenario plots.
                Defaults to None, meaning to plots are generated.
        """
        self.dataset_root = dataset_root
        self.plot_dir = plot_dir

    def compute_lane_miss_rate(
        self,
        forecasted_trajectories: Dict[int, NDArray[np.float64]],
        gt_trajectories: Dict[int, NDArray[np.float64]],
        velocity_treshold_factor: float = VELOCITY_TRHESHOLD_FACTOR,
        velocity_threshold_offset: float = VELOCITY_TRHESHOLD_OFFSET,
        n_cpus: int = 1,
    ) -> Dict[str, float]:
        """Calculate the Lane Miss Rate (LMR) for k=1 and multi-modal predictions.
            The results are accumulated over all given scenarios.

        Args:
            forecasted_trajectories (Dict[int, NDArray[np.float64]]): Sequence ids
                and predicted trajectories in the format
                [Number of Modes, Number of Timesteps, 2 (x and y coordinates)].
            gt_trajectories (Dict[int, NDArray[np.float64]]): Sequence ids and
                ground-truth trajectory in the format
                [Number of Timesteps, 2 (x and y coordinates)].
            velocity_treshold_factor (float, optional): Factor to determine
                hit threshold. Defaults to VELOCITY_TRHESHOLD_FACTOR.
            velocity_threshold_offset (float, optional): Constant offset to determine
                hit threshold. Defaults to VELOCITY_TRHESHOLD_OFFSET.
            n_cpus (int, optional): Number of CPUs for parallel processing.
                Defaults to 1.

        Returns:
            Dict[str, float]: Lane Miss Rate (LMR).
        """
        input_values = [
            (
                forecasted_trajectories[key],
                gt_trajectories[key],
                key,
                velocity_treshold_factor,
                velocity_threshold_offset,
            )
            for key in forecasted_trajectories
        ]

        # Parallel computation of hits and misses, returning values
        # for the top 1 prediction (k=1) and all predictions
        is_miss = process_map(
            self._compute_is_missed_lane_wrapper, input_values, max_workers=n_cpus
        )
        is_miss = np.stack(is_miss)

        # Average misses over the whole dataset
        metrics = dict()
        metrics["lmr_k1"] = is_miss[:, 0].mean()
        metrics["lmr"] = is_miss.all(axis=-1).mean()

        return metrics

    def _compute_is_missed_lane_wrapper(self, args: tuple) -> NDArray[np.bool_]:
        """Wrapper method to enable parallel processing of scenarios.

        Args:
            args (tuple): Input values.

        Returns:
            NDArray[np.bool_]: Boolean flag that indicates for each predicted
                trajectory whether it is a hit (False) or a miss (True).
        """
        return self.compute_is_missed_lane(*args)

    def compute_is_missed_lane(
        self,
        forecasted_trajectories: NDArray[np.float64],
        gt_trajectory: NDArray[np.float64],
        argo_id: str,
        velocity_treshold_factor: float = VELOCITY_TRHESHOLD_FACTOR,
        velocity_threshold_offset: float = VELOCITY_TRHESHOLD_OFFSET,
    ) -> NDArray[np.bool_]:
        """Caluclate whether predictions are a hit or a
            miss for a single scenario.

        Args:
            forecasted_trajectories (NDArray[np.float64]): Predicted trajectories
                in the format
                [Number of Modes, Number of Timesteps, 2 (x and y coordinates)].
            gt_trajectory (NDArray[np.float64]): Ground-truth trajectory
                in the format [Number of Timesteps, 2 (x and y coordinates)].
            argo_id (str): Argoverse 2 id of the current scenario.
            velocity_treshold_factor (float, optional): Factor to determine
                hit threshold. Defaults to VELOCITY_TRHESHOLD_FACTOR.
            velocity_threshold_offset (float, optional): Constant offset to
                determine hit threshold. Defaults to VELOCITY_TRHESHOLD_OFFSET.

        Returns:
            NDArray[np.bool_]: Boolean flag that indicates for each predicted
                trajectory whether it is a hit (False) or a miss (True).
        """
        # Load the map of the current scenario
        scenario_root = os.path.join(self.dataset_root, argo_id)
        static_map_path = os.path.join(scenario_root, f"log_map_archive_{argo_id}.json")
        scenario_map = ArgoverseStaticMap.from_json(Path(static_map_path))

        # Get centerlines
        lane_segments = scenario_map.vector_lane_segments
        scenario_lane_centerlines = {
            lane_seg_id: interp_utils.compute_midpoint_line(
                left_ln_boundary=lane_segment.left_lane_boundary.xyz,
                right_ln_boundary=lane_segment.right_lane_boundary.xyz,
                num_interp_pts=interp_utils.NUM_CENTERLINE_INTERP_PTS,
            )
            for lane_seg_id, lane_segment in lane_segments.items()
        }

        # Initialize R-tree
        rtree = self._get_rtree(scenario_lane_centerlines)

        # Calculate average velocity of ground-truth trajectory
        gt_deltas = gt_trajectory[1:] - gt_trajectory[:-1]
        gt_velocity_per_ts = (
            np.sqrt(gt_deltas[:, 0] ** 2 + gt_deltas[:, 1] ** 2)
            * DATASET_SAMPLING_FREQUENCY
        )
        gt_velocity = np.mean(gt_velocity_per_ts)

        # Calculate velocity-dependent lane hit threshold s_hit
        s_val_threshold = (
            velocity_treshold_factor * gt_velocity
        ) + velocity_threshold_offset

        # Get assignment for the ground-truth
        gt_orientation = np.arctan2(
            gt_trajectory[-1, 1] - gt_trajectory[-2, 1],
            gt_trajectory[-1, 0] - gt_trajectory[-2, 0],
        )
        gt_assignment = self._get_lane_assignments(
            scenario_lane_centerlines, rtree, gt_trajectory[-1], gt_orientation
        )

        # Check if there is no ground-truth assignment
        # We then fall back to original Miss Rate but use
        # the dynamic threshold we defined
        if len(gt_assignment) == 0:
            distance_to_gt = np.sqrt(
                (
                    (
                        forecasted_trajectories[:, -1]
                        - np.expand_dims(gt_trajectory[-1], 0)
                    )
                    ** 2
                ).sum(axis=-1)
            )
            return distance_to_gt > s_val_threshold

        # Extract assignment with highest confidence value
        max_gt_assignment = max(gt_assignment, key=attrgetter("confidence"))

        forecasted_assignments = []
        for forecasted_trajectory in forecasted_trajectories:
            # Get assignments for the prediction
            forecasted_orientation = np.arctan2(
                forecasted_trajectory[-1, 1] - forecasted_trajectory[-2, 1],
                forecasted_trajectory[-1, 0] - forecasted_trajectory[-2, 0],
            )
            curr_forecasted_assignments = self._get_lane_assignments(
                scenario_lane_centerlines,
                rtree,
                forecasted_trajectory[-1],
                forecasted_orientation,
            )

            if len(curr_forecasted_assignments) >= 1:
                # Extract assignment with highest confidence value
                max_forecasted_assignment = max(
                    curr_forecasted_assignments, key=attrgetter("confidence")
                )

                # Extract all assignments within threshold of assignment confidence
                max_forecasted_assignment_confidence = (
                    max_forecasted_assignment.confidence
                )
                curr_forecasted_assignments = [
                    x
                    for x in curr_forecasted_assignments
                    if x.confidence
                    >= max_forecasted_assignment_confidence
                    - ASSIGNMENT_CONFIDENCE_THRESHOLD
                ]

                # Sort remaining assignments by confidence (highest first)
                curr_forecasted_assignments = sorted(
                    curr_forecasted_assignments,
                    key=attrgetter("confidence"),
                    reverse=True,
                )

                # Take the highest confidence and remove all
                # others that assign to succ or prec lane segments
                curr_forecasted_assignments_clean = curr_forecasted_assignments
                for curr_forecasted_assignment in curr_forecasted_assignments:
                    # Check if curr_forecasted_assignmend is in
                    # curr_forecasted_assignments_clean
                    # If this is the case, remove all with lower confidence
                    if curr_forecasted_assignment in curr_forecasted_assignments_clean:
                        # Get index in curr_forecasted_assignments_clean
                        idx = curr_forecasted_assignments_clean.index(
                            curr_forecasted_assignment
                        )

                        successing_ids = scenario_map.vector_lane_segments[
                            curr_forecasted_assignment.lane_seg_id
                        ].successors
                        preceding_ids = scenario_map.vector_lane_segments[
                            curr_forecasted_assignment.lane_seg_id
                        ].predecessors
                        combined_ids = successing_ids + preceding_ids

                        idx_of_ids_to_delete = set()
                        # Check all remaining ones (with higher numbers than the current index)
                        # if they are they are directly successing or preceding wrt
                        # to the current lane segment
                        for i in range(idx + 1, len(curr_forecasted_assignments_clean)):
                            id_to_check = curr_forecasted_assignments_clean[
                                i
                            ].lane_seg_id
                            if id_to_check in combined_ids:
                                idx_of_ids_to_delete.add(i)

                        # Remove the assignments
                        curr_forecasted_assignments_clean = [
                            i
                            for j, i in enumerate(curr_forecasted_assignments_clean)
                            if j not in list(idx_of_ids_to_delete)
                        ]
            else:
                curr_forecasted_assignments_clean = [None]
            forecasted_assignments.append(curr_forecasted_assignments_clean)

        # Get the lanes and the s values on these lanes that are reachable
        # from the ground-truth assignment via the s_val_treshold
        reachable_lanes = self._get_reachable_lanes(
            max_gt_assignment.lane_seg_id,
            max_gt_assignment.s_val,
            s_val_threshold,
            scenario_map,
        )

        is_missed = []
        # Check if predictions are reachable from the ground-truth assignment
        # and lie within the s_val_treshold
        for curr_forecasted_assignments in forecasted_assignments:
            curr_is_missed = []
            for forecasted_assignment in curr_forecasted_assignments:
                if forecasted_assignment is None:
                    curr_is_missed.append(True)
                elif forecasted_assignment.lane_seg_id not in reachable_lanes:
                    curr_is_missed.append(True)
                else:
                    # Prediction and ground-truth are on the same lane segment
                    if (
                        forecasted_assignment.lane_seg_id
                        == max_gt_assignment.lane_seg_id
                    ):
                        # Prediction is in front of ground-truth
                        if forecasted_assignment.s_val > max_gt_assignment.s_val:
                            if (
                                reachable_lanes[max_gt_assignment.lane_seg_id][
                                    "forward"
                                ]
                                == -1
                                or reachable_lanes[max_gt_assignment.lane_seg_id][
                                    "forward"
                                ]
                                >= forecasted_assignment.s_val
                            ):
                                curr_is_missed.append(False)
                            else:
                                curr_is_missed.append(True)
                        else:
                            if (
                                reachable_lanes[max_gt_assignment.lane_seg_id][
                                    "backward"
                                ]
                                == -1
                                or reachable_lanes[max_gt_assignment.lane_seg_id][
                                    "backward"
                                ]
                                <= forecasted_assignment.s_val
                            ):
                                curr_is_missed.append(False)
                            else:
                                curr_is_missed.append(True)
                    else:
                        # Whole segment is reachable in forward direction
                        if (
                            "forward"
                            in reachable_lanes[forecasted_assignment.lane_seg_id]
                            and reachable_lanes[forecasted_assignment.lane_seg_id][
                                "forward"
                            ]
                            == -1
                        ):
                            curr_is_missed.append(False)
                        # Whole segment is reachable in backward direction
                        elif (
                            "backward"
                            in reachable_lanes[forecasted_assignment.lane_seg_id]
                            and reachable_lanes[forecasted_assignment.lane_seg_id][
                                "backward"
                            ]
                            == -1
                        ):
                            curr_is_missed.append(False)
                        # Prediction assignment is reachable in forward direction
                        elif (
                            "forward"
                            in reachable_lanes[forecasted_assignment.lane_seg_id]
                            and forecasted_assignment.s_val
                            <= reachable_lanes[forecasted_assignment.lane_seg_id][
                                "forward"
                            ]
                        ):
                            curr_is_missed.append(False)
                        # Prediction assignment is reachable in backward direction
                        elif (
                            "backward"
                            in reachable_lanes[forecasted_assignment.lane_seg_id]
                            and forecasted_assignment.s_val
                            >= reachable_lanes[forecasted_assignment.lane_seg_id][
                                "backward"
                            ]
                        ):
                            curr_is_missed.append(False)
                        # Not reachable
                        else:
                            curr_is_missed.append(True)
            # Current prediction only is a miss, if all assignments result in a miss
            is_missed.append(all(curr_is_missed))

        is_missed = np.array(is_missed)

        if self.plot_dir is not None:
            scenario = scenario_serialization.load_argoverse_scenario_parquet(
                os.path.join(scenario_root, f"scenario_{argo_id}.parquet")
            )
            save_path = os.path.join(self.plot_dir, f"{argo_id}.pdf")

            visualize_scenario(
                scenario,
                scenario_map,
                gt_trajectory=gt_trajectory,
                gt_assignment=max_gt_assignment,
                reachable_lanes=reachable_lanes,
                forecasted_trajectories=forecasted_trajectories,
                forecasted_assignments=forecasted_assignments,
                is_missed=is_missed,
                save_path=save_path,
                fixed_plot_size=FIXED_PLOT_SIZE,
            )

        return is_missed

    def _get_lane_assignments(
        self,
        scenario_lane_centerlines: Dict[int, Tuple[NDArray[np.float64], float]],
        rtree: index.Index,
        query_xy: NDArray[np.float64],
        query_orientation: float = None,
        query_radius: float = 3.0,  # width of a US freeway is 12 feet ~= 4m
    ) -> List[LaneAssignment]:
        """Assign a trajectory to one ore multiple lane segment centerlines.

        Args:
            scenario_lane_centerlines (Dict[int, Tuple[NDArray[np.float64], float]]):
                Lane segment ids and tuples containing centerline
                coordinates and lane widths.
            rtree (index.Index): R-tree of lane segments.
            query_xy (NDArray[np.float64]): Query location, in this
                case a trajectory endpoint.
            query_orientation (float, optional): Query orientation, in this case the
                orientation of the trajectory at the endpoint. Defaults to None.
            query_radius (float, optional): Query radius around the endpoint.
                Defaults to 3.0.

        Returns:
            List[LaneAssignment]: List of lane assignments.
        """
        # Use R-tree to get nearby lane segments
        query_box = (
            query_xy[0] - query_radius,
            query_xy[1] - query_radius,
            query_xy[0] + query_radius,
            query_xy[1] + query_radius,
        )
        box_hits = rtree.intersection(query_box, objects="raw")

        query_point = Point(query_xy)

        lane_assignements = []
        unique_lane_ids = []
        for box_hit in box_hits:
            lane_seg_id = box_hit[0]

            # Only check each lane segment once (the R-tree can result in
            # multiple matches per segment)
            if lane_seg_id in unique_lane_ids:
                continue
            else:
                unique_lane_ids.append(lane_seg_id)

            line = LineString(scenario_lane_centerlines[lane_seg_id][0][:, :2])
            line_len = line.length
            s_val = line.project(query_point)
            matched_point = line.interpolate(s_val)
            d_val = np.sqrt(
                (query_point.x - matched_point.x) ** 2
                + (query_point.y - matched_point.y) ** 2
            )

            # Check if the d_val is bigger than lane width / 2.0 (assignment is
            # outside of lane boundaries)
            if d_val > (scenario_lane_centerlines[lane_seg_id][1] / 2.0):
                continue

            # Calculate distance-based assignment confidence
            assignment_confidence = max(0.0, 1.0 - (d_val / DISTANCE_THRESHOLD))

            if query_orientation is not None:
                # Calculate angle in both directions
                orientation_forward = None
                orientation_backward = None

                # Approximate orientation of lane segment by going forward and
                # backward from the assigned point
                # If the assigned point is the start or end of a lane segment,
                # only one direction is possible
                if s_val >= ORIENTATION_ESTIMATION_STEP_SIZE:
                    point_backward = line.interpolate(
                        s_val - ORIENTATION_ESTIMATION_STEP_SIZE
                    )
                    orientation_backward = np.arctan2(
                        matched_point.y - point_backward.y,
                        matched_point.x - point_backward.x,
                    )
                if s_val <= (line_len - ORIENTATION_ESTIMATION_STEP_SIZE):
                    point_forward = line.interpolate(
                        s_val + ORIENTATION_ESTIMATION_STEP_SIZE
                    )
                    orientation_forward = np.arctan2(
                        point_forward.y - matched_point.y,
                        point_forward.x - matched_point.x,
                    )
                if orientation_forward is None and orientation_backward is None:
                    print(
                        "Matched a lane segment that is too short to calculate an orientation"
                    )
                    continue

                orientations = [
                    x for x in [orientation_backward, orientation_forward] if x != None
                ]
                average_orientation = sum(orientations) / len(orientations)

                # Get orientation difference between lane and trajectory
                orientation_difference = abs(
                    np.arctan2(
                        np.sin(query_orientation - average_orientation),
                        np.cos(query_orientation - average_orientation),
                    )
                )

                # Get combined assignment confidence
                orientation_confidence = max(
                    0.0, 1.0 - (orientation_difference / ORIENTATION_THRESHOLD)
                )
                assignment_confidence = (
                    (1.0 - ORIENTATION_WEIGHTING) * assignment_confidence
                    + ORIENTATION_WEIGHTING * orientation_confidence
                )

            lane_assignements.append(
                LaneAssignment(
                    lane_seg_id,
                    assignment_confidence,
                    s_val,
                    d_val,
                    line_len,
                    [matched_point.x, matched_point.y],
                )
            )

        return lane_assignements

    def _get_rtree(
        self,
        scenario_lane_centerlines: Dict[int, Tuple[NDArray[np.float64], float]],
    ) -> index.Index:
        """Build an R-tree of lane centerline segments.

        Args:
            scenario_lane_centerlines (Dict[int, Tuple[NDArray[np.float64], float]]):
                Lane segment ids and tuples containing centerline
                coordinates and lane widths.

        Returns:
            index.Index: R-tree.
        """

        def generate_items():
            subseg_id = 0
            for lane_seg_id, lane_segment in scenario_lane_centerlines.items():
                # Only extract x and y, throw z away
                lane_seg_coords = lane_segment[0][:, :2]
                for i in range(len(lane_seg_coords) - 1):
                    x_1, y_1 = lane_seg_coords[i]
                    x_2, y_2 = lane_seg_coords[i + 1]
                    subseg = ((x_1, y_1), (x_2, y_2))
                    subseg_box = (
                        min(x_1, x_2),
                        min(y_1, y_2),
                        max(x_1, x_2),
                        max(y_1, y_2),
                    )
                    yield (subseg_id, subseg_box, (lane_seg_id, subseg))
                    subseg_id += 1

        return index.Index(generate_items())

    def _get_reachable_lanes(
        self,
        start_id: int,
        s_val_on_start: float,
        s_val_threshold: float,
        scenario_map: ArgoverseStaticMap,
    ) -> Dict[int, Dict[str, float]]:
        """Get each lane segment and also the point on each corresponding
            lane segment that is reachable from a starting point on an
            initial lane segment. Only reachable lane segments that are
            within a threshold distance along the lane graph are returned.
            Lane segments can be reachable up to a Frenet s-value in
            "forward" and "backward" direction.

        Args:
            start_id (int): Id of the initial lane segment.
            s_val_on_start (float): Frenet s-value of starting point on
                the initial lane segment.
            s_val_threshold (float): Threshold distance along the lane graph.
            scenario_map (ArgoverseStaticMap): Scenario map api of Argoverse 2.

        Returns:
            Dict[int, Dict[str, float]]: Dictionary consisting of the lane
                segment id and dictionary pairs.
                The second-order dictionary holds the reachability of the
                corresponding lane segment:
                A float value assigned to the "forward" key indicatates that
                the lane segment is reachable up to the Frenet s-value
                in forward direction.
                A float value assigned to the "backward" key indicatates that
                the lane segment is reachable up to the Frenet s-value
                in backward direction.
                In both cases the value -1 indicates that the whole lane
                segment is reachable.
        """
        centerline = scenario_map.get_lane_segment_centerline(start_id)
        centerline_length = LineString(centerline).length

        # Perform dfs in forward and backward direction
        # starting from the assigned lane segment
        visited = {}
        dfs_result_forward = self._dfs(
            visited=visited,
            direction="forward",
            start_id=start_id,
            s_val_on_start=s_val_on_start,
            start_centerline_length=centerline_length,
            s_val_traveled=centerline_length - s_val_on_start,
            s_val_threshold=s_val_threshold,
            scenario_map=scenario_map,
        )

        visited = {}
        dfs_result_backward = self._dfs(
            visited=visited,
            direction="backward",
            start_id=start_id,
            s_val_on_start=s_val_on_start,
            start_centerline_length=centerline_length,
            s_val_traveled=s_val_on_start,
            s_val_threshold=s_val_threshold,
            scenario_map=scenario_map,
        )

        dfs_result = dfs_result_forward + dfs_result_backward

        # It might be possible to visit some lane segments multiple times (loops)
        # We then only keep the one segment and the reachable position that is further
        # away from the starting point
        reachable_lanes = dict()
        for dfs_sequence in dfs_result:
            for dfs_sequence_entry in dfs_sequence:
                if dfs_sequence_entry[0] not in reachable_lanes:
                    reachable_lanes[dfs_sequence_entry[0]] = dict()

                if dfs_sequence_entry[1] == "forward":
                    if "forward" not in reachable_lanes[dfs_sequence_entry[0]]:
                        reachable_lanes[dfs_sequence_entry[0]][
                            "forward"
                        ] = dfs_sequence_entry[2]
                    elif dfs_sequence_entry[2] == -1:
                        reachable_lanes[dfs_sequence_entry[0]]["forward"] = -1
                    else:
                        reachable_lanes[dfs_sequence_entry[0]]["forward"] = max(
                            reachable_lanes[dfs_sequence_entry[0]]["forward"],
                            dfs_sequence_entry[2],
                        )

                if dfs_sequence_entry[1] == "backward":
                    if "backward" not in reachable_lanes[dfs_sequence_entry[0]]:
                        reachable_lanes[dfs_sequence_entry[0]][
                            "backward"
                        ] = dfs_sequence_entry[2]
                    elif dfs_sequence_entry[2] == -1:
                        reachable_lanes[dfs_sequence_entry[0]]["backward"] = -1
                    else:
                        reachable_lanes[dfs_sequence_entry[0]]["backward"] = min(
                            reachable_lanes[dfs_sequence_entry[0]]["backward"],
                            dfs_sequence_entry[2],
                        )

        return reachable_lanes

    def _dfs(
        self,
        visited: Dict[int, Dict[str, bool]],
        direction: str,
        start_id: int,
        s_val_on_start: float,
        start_centerline_length: float,
        s_val_traveled: float,
        s_val_threshold: float,
        scenario_map: ArgoverseStaticMap,
    ) -> List[List[Tuple[int, str, float]]]:
        """Directed depth-first search that is able to go into
            forward or backward direction.

        Args:
            visited (Dict[int, Dict[str, bool]]): Dictionary consisting of lane
                segment id and dictionary pairs. The second-order dictionary
                holds the direction ("forward" or "backward") of visited
                lane segments.
            direction (str): Search direction ("forward" or "backward").
            start_id (int): Id of the initial lane segment.
            s_val_on_start (float): Frenet s-value of starting point on the
                initial lane segment.
            start_centerline_length (float): Centerline length of the initial
                lane segment.
            s_val_traveled (float): Already travelend distance along the lane
                graph.
            s_val_threshold (float): Threshold distance along the lane graph.
            scenario_map (ArgoverseStaticMap): Scenario map api of Argoverse 2.

        Returns:
            List[List[Tuple[int, str, float]]]: List containing the results of
                the depth-first search. Each list entry contains one possible
                path. Each path is defined by edges. These edges consist of:
                    int: Id of the lane segment.
                    str: Traversed direction during the search ("forward" or
                        "backward").
                    float: Reachable Frenet s-value in search direction,
                        with -1 indicating full reachability.
        """
        # Check if node already got visited from both directions
        if start_id in visited:
            if "forward" in visited[start_id] and "backward" in visited[start_id]:
                if visited[start_id]["forward"] and visited[start_id]["backward"]:
                    return [[]]
        else:
            visited[start_id] = dict()

        # Set the visited flags
        if direction == "forward":
            visited[start_id]["forward"] = True

        if direction == "backward":
            visited[start_id]["backward"] = True

        # Check if the threshold got exceeded and return if so
        if s_val_traveled > s_val_threshold:
            if direction == "forward":
                traveled_on_current_lane_segment = (
                    start_centerline_length - s_val_on_start
                )
                traveled_without_current_lane_segment = (
                    s_val_traveled - traveled_on_current_lane_segment
                )
                left_distance_on_current_lane_segment = (
                    s_val_threshold - traveled_without_current_lane_segment
                )
                s_on_lane = s_val_on_start + left_distance_on_current_lane_segment

            if direction == "backward":
                traveled_on_current_lane_segment = s_val_on_start
                traveled_without_current_lane_segment = (
                    s_val_traveled - traveled_on_current_lane_segment
                )
                left_distance_on_current_lane_segment = (
                    s_val_threshold - traveled_without_current_lane_segment
                )
                s_on_lane = s_val_on_start - left_distance_on_current_lane_segment

            return [[(start_id, direction, s_on_lane)]]

        curr_path = []

        # If direction is forward, another dfs for all successors is
        # triggered in forward direction
        # Also, for each predecessor of each successor, a dfs in backward
        # direction is triggered
        if direction == "forward":
            successing_ids = scenario_map.vector_lane_segments[start_id].successors
            for successing_id in successing_ids:
                if successing_id in scenario_map.vector_lane_segments:

                    successing_centerline = scenario_map.get_lane_segment_centerline(
                        successing_id
                    )
                    successing_centerline_length = LineString(
                        successing_centerline
                    ).length

                    dfs_result = self._dfs(
                        visited=visited,
                        direction="forward",
                        start_id=successing_id,
                        s_val_on_start=0,
                        start_centerline_length=successing_centerline_length,
                        s_val_traveled=s_val_traveled
                        + successing_centerline_length
                        - 0,
                        s_val_threshold=s_val_threshold,
                        scenario_map=scenario_map,
                    )

                    curr_path.extend(dfs_result)

                    predecessor_ids_of_successor = scenario_map.vector_lane_segments[
                        successing_id
                    ].predecessors
                    for predecessor_id_of_successor in predecessor_ids_of_successor:
                        if (
                            predecessor_id_of_successor
                            in scenario_map.vector_lane_segments
                            and predecessor_id_of_successor != start_id
                        ):

                            centerline = scenario_map.get_lane_segment_centerline(
                                predecessor_id_of_successor
                            )
                            centerline_length = LineString(centerline).length

                            dfs_result = self._dfs(
                                visited=visited,
                                direction="backward",
                                start_id=predecessor_id_of_successor,
                                s_val_on_start=centerline_length,
                                start_centerline_length=centerline_length,
                                s_val_traveled=s_val_traveled + centerline_length,
                                s_val_threshold=s_val_threshold,
                                scenario_map=scenario_map,
                            )

                            curr_path.extend(dfs_result)

        # If direction is backward, another dfs for all predecessors is
        # triggered in backward direction
        # Also, for each successor of each predecessor, a dfs in forward
        # direction is triggered
        if direction == "backward":
            preceding_ids = scenario_map.vector_lane_segments[start_id].predecessors
            for preceding_id in preceding_ids:
                if preceding_id in scenario_map.vector_lane_segments:

                    preceding_centerline = scenario_map.get_lane_segment_centerline(
                        preceding_id
                    )
                    preceding_centerline_length = LineString(
                        preceding_centerline
                    ).length

                    dfs_result = self._dfs(
                        visited=visited,
                        direction="backward",
                        start_id=preceding_id,
                        s_val_on_start=preceding_centerline_length,
                        start_centerline_length=preceding_centerline_length,
                        s_val_traveled=s_val_traveled + preceding_centerline_length,
                        s_val_threshold=s_val_threshold,
                        scenario_map=scenario_map,
                    )

                    curr_path.extend(dfs_result)

                    successor_ids_of_predecessor = scenario_map.vector_lane_segments[
                        preceding_id
                    ].successors
                    for successor_id_of_predecessor in successor_ids_of_predecessor:
                        if (
                            successor_id_of_predecessor
                            in scenario_map.vector_lane_segments
                            and successor_id_of_predecessor != start_id
                        ):

                            centerline = scenario_map.get_lane_segment_centerline(
                                successor_id_of_predecessor
                            )
                            centerline_length = LineString(centerline).length

                            dfs_result = self._dfs(
                                visited=visited,
                                direction="forward",
                                start_id=successor_id_of_predecessor,
                                s_val_on_start=0,
                                start_centerline_length=centerline_length,
                                s_val_traveled=s_val_traveled + centerline_length,
                                s_val_threshold=s_val_threshold,
                                scenario_map=scenario_map,
                            )

                            curr_path.extend(dfs_result)

        if len(curr_path) == 0:
            return [[(start_id, direction, -1)]]

        return_value = []
        for curr_subpath in curr_path:
            return_value.append(curr_subpath + [(start_id, direction, -1)])

        return return_value
