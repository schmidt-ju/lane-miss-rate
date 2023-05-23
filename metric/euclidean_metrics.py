from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray


class EuclideanMetrics:
    def compute_euclidean_metrics(
        self,
        forecasted_trajectories: Dict[int, NDArray[np.float64]],
        gt_trajectories: Dict[int, NDArray[np.float64]],
    ) -> Dict[str, float]:
        """Calculate Euclidean metrics (Average Displacement Error (ADE), Final
            Displacement Error (FDE) and Miss Rate (MR)) for k=1 and multi-modal
            predictions. The results are accumulated over all given scenarios.

        Args:
            forecasted_trajectories (Dict[int, NDArray[np.float64]]):
                Sequence ids and predicted trajectories in the format
                [Number of Modes, Number of Timesteps, 2 (x and y coordinates)].
            gt_trajectories (Dict[int, NDArray[np.float64]]):
                Sequence ids and ground-truth trajectory in the format
                [Number of Timesteps, 2 (x and y coordinates)].

        Returns:
            Dict[str, float]: Euclidean metrics.
        """
        forecasted_trajectories = [
            forecasted_trajectories[key] for key in forecasted_trajectories
        ]
        gt_trajectories = [gt_trajectories[key] for key in gt_trajectories]

        ade1, fde1, mr1, ade, fde, mr, min_idcs = self._compute_euclidean_metrics(
            forecasted_trajectories, gt_trajectories
        )

        metrics = dict()
        metrics["ade_k1"] = ade1
        metrics["fde_k1"] = fde1
        metrics["mr_k1"] = mr1
        metrics["ade"] = ade
        metrics["fde"] = fde
        metrics["mr"] = mr

        return metrics

    def _compute_euclidean_metrics(
        self,
        forecasted_trajectories: List[NDArray[np.float64]],
        gt_trajectories: List[NDArray[np.float64]],
    ) -> Tuple[float, float, float, float, float, int]:
        """_summary_

        Args:
            forecasted_trajectories (List[NDArray[np.float64]]):
                List of predicted trajectories in the format
                [Number of Modes, Number of Timesteps, 2 (x and y coordinates)].
            gt_trajectories (List[NDArray[np.float64]]):
                List of ground-truth trajectories in the format
                [Number of Timesteps, 2 (x and y coordinates)].

        Returns:
            Tuple[float, float, float, float, float, int]:
                Metrics and best mode indices.
        """
        preds = np.array(forecasted_trajectories, dtype=np.float32)
        gt_preds = np.array(gt_trajectories, dtype=np.float32)

        # Compute timestep-wise Euclidean error
        err = np.sqrt(np.sum((preds - np.expand_dims(gt_preds, axis=1)) ** 2, axis=3))

        # Compute metrics for the top-ranked mode (index=0)
        ade1 = np.mean(err[:, 0])
        fde1 = np.mean(err[:, 0, -1])
        mr1_bool = err[:, 0, -1] > 2.0
        mr1 = np.sum(mr1_bool) / len(mr1_bool)

        # Compute metrics for the mode with the smallest endpoint error
        min_idcs = np.argmin(err[:, :, -1], axis=1)
        row_idcs = np.arange(len(min_idcs)).astype(np.int64)
        err = err[row_idcs, min_idcs]
        ade = np.mean(err)
        fde = np.mean(err[:, -1])
        mr_bool = err[:, -1] > 2.0
        mr = np.sum(mr_bool) / len(mr_bool)

        return ade1, fde1, mr1, ade, fde, mr, min_idcs
