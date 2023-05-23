import argparse
import os
import random

import numpy as np
import pandas as pd

from metric.euclidean_metrics import EuclideanMetrics
from metric.lane_miss_rate import LaneMissRate


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_root",
    type=str,
    default=os.path.join(
        os.path.realpath(os.path.dirname(__file__)), "data", "val"
    ),
)
parser.add_argument(
    "--plot_dir",
    type=str,
    default=os.path.join(
        os.path.realpath(os.path.dirname(__file__)), "viz", "plots"
    ),
)
parser.add_argument(
    "--scenario_id", type=str, default="00062a32-8d6d-4449-9948-6fedac67bfcd"
)
parser.add_argument("--n_preds", type=int, default=6)


def sample_scenario():
    """Evaluate and plot a sample scenario of the Argoverse 2 dataset using
        Lane Miss Rate (LMR) and Euclidean metrics."""

    args = parser.parse_args()

    # Init metrics
    euclidean_metrics = EuclideanMetrics()
    lane_miss_rate = LaneMissRate(
        dataset_root=args.dataset_root, plot_dir=args.plot_dir
    )

    # Read scenario
    df = pd.read_parquet(
        os.path.join(
            args.dataset_root, args.scenario_id, f"scenario_{args.scenario_id}.parquet"
        )
    )
    # 3 is the focal agent
    df = df[df["object_category"] == 3]

    # Check trajectory length of the focal agent
    if len(df.index) != 110:
        print("Trajectory is too short")
        return

    # Check object type of the focal agent
    if df["object_type"].iloc[0] not in ["vehicle", "motorcyclist", "bus"]:
        print("Class is", df["object_type"].iloc[0])
        return

    # Extract trajectory
    traj_full = np.concatenate(
        (
            df.position_x.to_numpy().reshape(-1, 1),
            df.position_y.to_numpy().reshape(-1, 1),
        ),
        1,
    )

    example_gt = traj_full[50:]

    # Generate dummy predictions that are randomly distributed
    # around the ground-truth endpoint
    example_pred = []
    for i in range(args.n_preds):
        curr_x_dev = random.uniform(-5, 5)
        curr_y_dev = random.uniform(-5, 5)

        values_to_interpolate = np.arange(0, 60)

        x = [traj_full[49, 0], traj_full[-1, 0] + curr_x_dev]
        y = [traj_full[49, 1], traj_full[-1, 1] + curr_y_dev]

        x_new = np.interp(values_to_interpolate, [0, 59], x)
        y_new = np.interp(values_to_interpolate, [0, 59], y)

        example_pred.append(np.stack([x_new, y_new]).swapaxes(0, 1))

    example_pred = np.array(example_pred)

    # Compute metrics
    print(
        euclidean_metrics.compute_euclidean_metrics(
            {args.scenario_id: example_pred}, {args.scenario_id: example_gt}
        )
    )
    print(
        lane_miss_rate.compute_lane_miss_rate(
            {args.scenario_id: example_pred}, {args.scenario_id: example_gt}
        )
    )


if __name__ == "__main__":
    sample_scenario()
