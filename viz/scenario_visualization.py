# This code is based on the official Argoverse 2 api:
# https://github.com/argoverse/av2-api/blob/main/src/av2/datasets/motion_forecasting/viz/scenario_visualization.py

# Modified on 24.03.2023:
# The original plotting logic got adapted to only output one plot and not a video.
# The original plotting logic got extended by centerlines, predictions, ground-truth,
# assignments of trajectories to centerlines and the Euclidean miss indicator.

# Argoverse 2 api is licensed under MIT.
# The full license is available in: LICENSE_ARGOVERSE2

"""Visualization utils for Argoverse MF scenarios."""

import math
from pathlib import Path
from typing import Dict, Final, List, Optional, Sequence, Set, Tuple

import av2.geometry.interpolate as interp_utils

import matplotlib.pyplot as plt
import numpy as np

from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario,
    ObjectType,
    TrackCategory,
)
from av2.map.map_api import ArgoverseStaticMap
from av2.utils.typing import NDArrayFloat, NDArrayInt
from matplotlib.patches import Rectangle
from numpy.typing import NDArray
from shapely.geometry import LineString

from metric.lane_assignment import LaneAssignment

_PlotBounds = Tuple[float, float, float, float]

# Configure constants
_OBS_DURATION_TIMESTEPS: Final[int] = 50
_PRED_DURATION_TIMESTEPS: Final[int] = 60

_ESTIMATED_VEHICLE_LENGTH_M: Final[float] = 4.0
_ESTIMATED_VEHICLE_WIDTH_M: Final[float] = 2.0
_ESTIMATED_CYCLIST_LENGTH_M: Final[float] = 2.0
_ESTIMATED_CYCLIST_WIDTH_M: Final[float] = 0.7
_PLOT_BOUNDS_BUFFER_M: Final[float] = 30.0

_DRIVABLE_AREA_COLOR: Final[str] = "#7A7A7A"
_LANE_SEGMENT_COLOR: Final[str] = "#E0E0E0"

_DEFAULT_ACTOR_COLOR: Final[str] = "#D3E8EF"
_FOCAL_AGENT_COLOR: Final[str] = "#4040ff"
_AV_COLOR: Final[str] = "#007672"
_BOUNDING_BOX_ZORDER: Final[
    int
] = 100  # Ensure actor bounding boxes are plotted on top of all map elements

_PRED_TRAJ_COLOR: Final[str] = "#d33e4c"
_PRED_ENDPOINT_COLOR_HIT: Final[str] = "#ffa500"
_PRED_ENDPOINT_COLOR_MISS: Final[str] = "#d33e4c"
_GT_TRAJ_COLOR: Final[str] = "#4cd33e"
_LANE_ASSIGNMENT_COLOR: Final[str] = "#CC00CC"

_STATIC_OBJECT_TYPES: Set[ObjectType] = {
    ObjectType.STATIC,
    ObjectType.BACKGROUND,
    ObjectType.CONSTRUCTION,
    ObjectType.RIDERLESS_BICYCLE,
}


def visualize_scenario(
    scenario: ArgoverseScenario,
    scenario_static_map: ArgoverseStaticMap,
    gt_trajectory: NDArray[np.float64],
    gt_assignment: LaneAssignment,
    reachable_lanes: Dict[int, Dict[str, float]],
    forecasted_trajectories: NDArray[np.float64],
    forecasted_assignments: List[List[LaneAssignment]],
    is_missed: NDArray[np.bool_],
    save_path: Path,
    fixed_plot_size: Tuple[float] = None,
) -> None:
    """Build visualization for the focal agent, its predictions, its ground-truth and
        the local map associated with the Argoverse scenario. This visualization also
        includes lane assignment of trajectories to lane centerlines.
    Args:
        scenario: Argoverse scenario to visualize.
        scenario_static_map: Local static map elements associated with `scenario`.
        gt_trajectory: Ground-truth trajectory of the focal agent.
        gt_assignment: Lane assignment of the ground-truth trajectory.
        reachable_lanes: Reachable lane segments that are within a threshold
            distance along the ground-truth lane assignment.
        forecasted_trajectories: Predicted trajectories of the focal agent.
        forecasted_assignments: Lane assignments of the predicted trajectories.
        is_missed: Indicator for each predicted trajectory whether it is a hit
            (False) or a miss (True) according to the Lane Miss Rate (LMR) definition.
        save_path: Path where plots should be saved.
        fixed_plot_size: Use fixed plot size or scale it dynamically with the scenario.
    """

    timestep = _OBS_DURATION_TIMESTEPS - 1
    _, ax = plt.subplots()

    # Plot static map elements and actor tracks
    _plot_static_map_elements(scenario_static_map, reachable_lanes)
    plot_bounds = _plot_actor_tracks(ax, scenario, timestep)

    # Plot predictions
    _plot_polylines(
        [x for x in forecasted_trajectories], color=_PRED_TRAJ_COLOR, line_width=1
    )
    for i, traj in enumerate(forecasted_trajectories):
        endpoint_color = (
            _PRED_ENDPOINT_COLOR_MISS if is_missed[i] else _PRED_ENDPOINT_COLOR_HIT
        )
        plt.plot(traj[-1, 0], traj[-1, 1], "*", color=endpoint_color, markersize=2)
        for forecasted_assignment in forecasted_assignments[i]:
            if forecasted_assignment is not None:
                plt.plot(
                    forecasted_assignment.matched_point[0],
                    forecasted_assignment.matched_point[1],
                    "x",
                    color=_LANE_ASSIGNMENT_COLOR,
                    markersize=1,
                )
                plt.plot(
                    [forecasted_assignment.matched_point[0], traj[-1, 0]],
                    [forecasted_assignment.matched_point[1], traj[-1, 1]],
                    "--",
                    linewidth=0.5,
                    color=_LANE_ASSIGNMENT_COLOR,
                )

    # Plot ground-truth
    _plot_polylines([gt_trajectory], color=_GT_TRAJ_COLOR, line_width=1)
    plt.plot(
        gt_trajectory[-1, 0],
        gt_trajectory[-1, 1],
        "*",
        color=_GT_TRAJ_COLOR,
        markersize=2,
    )
    plt.plot(
        gt_assignment.matched_point[0],
        gt_assignment.matched_point[1],
        "x",
        color=_LANE_ASSIGNMENT_COLOR,
        markersize=1,
    )
    plt.plot(
        [gt_assignment.matched_point[0], gt_trajectory[-1, 0]],
        [gt_assignment.matched_point[1], gt_trajectory[-1, 1]],
        "--",
        linewidth=0.5,
        color=_LANE_ASSIGNMENT_COLOR,
    )
    # Plot Euclidean MR circle around the ground-truth
    mr_circle = plt.Circle(
        (gt_trajectory[-1, 0], gt_trajectory[-1, 1]),
        2,
        color="#f59042",
        alpha=0.5,
        linewidth=0,
    )
    ax.add_patch(mr_circle)

    # Update min and max plotting bounds
    gt_and_preds = np.concatenate([forecasted_trajectories, [gt_trajectory]])
    gt_and_preds = np.concatenate(gt_and_preds)
    x_min, x_max = gt_and_preds[:, 0].min(), gt_and_preds[:, 0].max()
    y_min, y_max = gt_and_preds[:, 1].min(), gt_and_preds[:, 1].max()
    plot_bounds = (
        min(plot_bounds[0], x_min),
        max(plot_bounds[1], x_max),
        min(plot_bounds[2], y_min),
        max(plot_bounds[3], y_max),
    )

    if fixed_plot_size != None:
        # Set map bounds to a fixed distance. The center is a little bit into
        # the ground-truth (1.5s)
        plt_bounds_xmin = gt_trajectory[15, 0] - fixed_plot_size[0] / 2
        plt_bounds_xmax = gt_trajectory[15, 0] + fixed_plot_size[0] / 2
        plt_bounds_ymin = gt_trajectory[15, 1] + fixed_plot_size[1] / 2
        plt_bounds_ymax = gt_trajectory[15, 1] - fixed_plot_size[1] / 2
    else:
        # Set map bounds to capture focal trajectory history
        # (with fixed buffer in all directions)
        plt_bounds_xmin = plot_bounds[0] - _PLOT_BOUNDS_BUFFER_M
        plt_bounds_xmax = plot_bounds[1] + _PLOT_BOUNDS_BUFFER_M
        plt_bounds_ymin = plot_bounds[2] - _PLOT_BOUNDS_BUFFER_M
        plt_bounds_ymax = plot_bounds[3] + _PLOT_BOUNDS_BUFFER_M

    plt.xlim(plt_bounds_xmin, plt_bounds_xmax)
    plt.ylim(plt_bounds_ymin, plt_bounds_ymax)
    plt.gca().set_aspect("equal", adjustable="box")

    # Minimize plot margins and make axes invisible
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.savefig(save_path, dpi=700)


def _plot_static_map_elements(
    static_map: ArgoverseStaticMap,
    reachable_lanes: Dict[int, Dict[str, float]],
    show_ped_xings: bool = False,
) -> None:
    """Plot all static map elements associated with an Argoverse scenario.
    Args:
        static_map: Static map containing elements to be plotted.
        reachable_lanes: Reachable lane segments that are within a threshold
            distance along the ground-truth lane assignment.
        show_ped_xings: Configures whether pedestrian crossings should be plotted.
    """
    # Plot drivable areas
    for drivable_area in static_map.vector_drivable_areas.values():
        _plot_polygons([drivable_area.xyz], alpha=0.5, color=_DRIVABLE_AREA_COLOR)

    # Plot lane segments with centerlines
    for lane_segment in static_map.vector_lane_segments.values():
        _plot_polylines(
            [
                lane_segment.left_lane_boundary.xyz,
                lane_segment.right_lane_boundary.xyz,
            ],
            line_width=0.5,
            color=_LANE_SEGMENT_COLOR,
        )
        scenario_lane_centerline, _ = interp_utils.compute_midpoint_line(
            left_ln_boundary=lane_segment.left_lane_boundary.xyz,
            right_ln_boundary=lane_segment.right_lane_boundary.xyz,
            num_interp_pts=interp_utils.NUM_CENTERLINE_INTERP_PTS,
        )
        _plot_polylines(
            [
                scenario_lane_centerline,
            ],
            line_width=0.5,
            style="--",
            color=_LANE_SEGMENT_COLOR,
        )

        # NOTE Optionally add centerline ids as text labels
        # plt.text(scenario_lane_centerline[0, 0], scenario_lane_centerline[0, 1], lane_segment.id, fontsize=2)

    # Add the reachable lanes
    for key, value in reachable_lanes.items():
        lane_segment = static_map.vector_lane_segments[key]
        scenario_lane_centerline, _ = interp_utils.compute_midpoint_line(
            left_ln_boundary=lane_segment.left_lane_boundary.xyz,
            right_ln_boundary=lane_segment.right_lane_boundary.xyz,
            num_interp_pts=interp_utils.NUM_CENTERLINE_INTERP_PTS,
        )
        line = LineString(scenario_lane_centerline)

        if "forward" in value:
            if value["forward"] != -1:
                matched_point = line.interpolate(value["forward"])
                plt.plot(
                    matched_point.x, matched_point.y, "o", markersize=0.3, color="blue"
                )
        if "backward" in value:
            if value["backward"] != -1:
                matched_point = line.interpolate(value["backward"])
                plt.plot(
                    matched_point.x,
                    matched_point.y,
                    "o",
                    markersize=0.3,
                    color="yellow",
                )

    # Plot pedestrian crossings
    if show_ped_xings:
        for ped_xing in static_map.vector_pedestrian_crossings.values():
            _plot_polylines(
                [ped_xing.edge1.xyz, ped_xing.edge2.xyz],
                alpha=1.0,
                color=_LANE_SEGMENT_COLOR,
            )


def _plot_actor_tracks(
    ax: plt.Axes, scenario: ArgoverseScenario, timestep: int
) -> Optional[_PlotBounds]:
    """Plot all actor tracks (up to a particular time step) associated with an Argoverse scenario.
    Args:
        ax: Axes on which actor tracks should be plotted.
        scenario: Argoverse scenario for which to plot actor tracks.
        timestep: Tracks are plotted for all actor data up to the specified time step.
    Returns:
        track_bounds: (x_min, x_max, y_min, y_max) bounds for the extent of actor tracks.
    """
    track_bounds = None
    for track in scenario.tracks:
        # Get timesteps for which actor data is valid
        actor_timesteps: NDArrayInt = np.array(
            [
                object_state.timestep
                for object_state in track.object_states
                if object_state.timestep <= timestep
            ]
        )
        if actor_timesteps.shape[0] < 1 or actor_timesteps[-1] != timestep:
            continue

        # Get actor trajectory and heading history
        actor_trajectory: NDArrayFloat = np.array(
            [
                list(object_state.position)
                for object_state in track.object_states
                if object_state.timestep <= timestep
            ]
        )
        actor_headings: NDArrayFloat = np.array(
            [
                object_state.heading
                for object_state in track.object_states
                if object_state.timestep <= timestep
            ]
        )

        # Plot polyline for focal agent location history
        track_color = _DEFAULT_ACTOR_COLOR
        if track.category == TrackCategory.FOCAL_TRACK:
            x_min, x_max = actor_trajectory[:, 0].min(), actor_trajectory[:, 0].max()
            y_min, y_max = actor_trajectory[:, 1].min(), actor_trajectory[:, 1].max()
            track_bounds = (x_min, x_max, y_min, y_max)
            track_color = _FOCAL_AGENT_COLOR
            _plot_polylines([actor_trajectory], color=track_color, line_width=1)
        else:
            continue
        # NOTE The whole logik to plot other agents than the target agent is removed
        # elif track.track_id == "AV":
        #    track_color = _AV_COLOR
        # elif track.object_type in _STATIC_OBJECT_TYPES:
        #    continue

        # Plot bounding boxes for all vehicles and cyclists
        if track.object_type == ObjectType.VEHICLE:
            _plot_actor_bounding_box(
                ax,
                actor_trajectory[-1],
                actor_headings[-1],
                track_color,
                (_ESTIMATED_VEHICLE_LENGTH_M, _ESTIMATED_VEHICLE_WIDTH_M),
            )
        elif (
            track.object_type == ObjectType.CYCLIST
            or track.object_type == ObjectType.MOTORCYCLIST
        ):
            _plot_actor_bounding_box(
                ax,
                actor_trajectory[-1],
                actor_headings[-1],
                track_color,
                (_ESTIMATED_CYCLIST_LENGTH_M, _ESTIMATED_CYCLIST_WIDTH_M),
            )
        else:
            plt.plot(
                actor_trajectory[-1, 0],
                actor_trajectory[-1, 1],
                "o",
                color=track_color,
                markersize=4,
            )

    return track_bounds


def _plot_polylines(
    polylines: Sequence[NDArrayFloat],
    *,
    style: str = "-",
    line_width: float = 1.0,
    alpha: float = 1.0,
    color: str = "r",
) -> None:
    """Plot a group of polylines with the specified config.
    Args:
        polylines: Collection of (N, 2) polylines to plot.
        style: Style of the line to plot (e.g. `-` for solid, `--` for dashed)
        line_width: Desired width for the plotted lines.
        alpha: Desired alpha for the plotted lines.
        color: Desired color for the plotted lines.
    """
    for polyline in polylines:
        plt.plot(
            polyline[:, 0],
            polyline[:, 1],
            style,
            linewidth=line_width,
            color=color,
            alpha=alpha,
        )


def _plot_polygons(
    polygons: Sequence[NDArrayFloat], *, alpha: float = 1.0, color: str = "r"
) -> None:
    """Plot a group of filled polygons with the specified config.
    Args:
        polygons: Collection of polygons specified by (N,2) arrays of vertices.
        alpha: Desired alpha for the polygon fill.
        color: Desired color for the polygon.
    """
    for polygon in polygons:
        plt.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=alpha)


def _plot_actor_bounding_box(
    ax: plt.Axes,
    cur_location: NDArrayFloat,
    heading: float,
    color: str,
    bbox_size: Tuple[float, float],
) -> None:
    """Plot an actor bounding box centered on the actor's current location.
    Args:
        ax: Axes on which actor bounding box should be plotted.
        cur_location: Current location of the actor (2,).
        heading: Current heading of the actor (in radians).
        color: Desired color for the bounding box.
        bbox_size: Desired size for the bounding box (length, width).
    """
    (bbox_length, bbox_width) = bbox_size

    # Compute coordinate for pivot point of bounding box
    d = np.hypot(bbox_length, bbox_width)
    theta_2 = math.atan2(bbox_width, bbox_length)
    pivot_x = cur_location[0] - (d / 2) * math.cos(heading + theta_2)
    pivot_y = cur_location[1] - (d / 2) * math.sin(heading + theta_2)

    vehicle_bounding_box = Rectangle(
        (pivot_x, pivot_y),
        bbox_length,
        bbox_width,
        np.degrees(heading),
        color=color,
        zorder=_BOUNDING_BOX_ZORDER,
    )
    ax.add_patch(vehicle_bounding_box)
