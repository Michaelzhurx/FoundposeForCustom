# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

from typing import Any, Dict, Tuple

import cv2
import numpy as np

from utils import (
    logging,
    misc
)
from utils.misc import tensor_to_array

from utils.structs import PinholePlaneCameraModel

logger: logging.Logger = logging.get_logger()


def _value_to_float(value: Any, default: float = 0.0) -> float:
    """Converts tensor/array/scalar values to a plain Python float."""

    if value is None:
        return default
    try:
        value = tensor_to_array(value)
    except Exception:
        pass

    value_np = np.asarray(value)
    if value_np.size == 0:
        return default

    return float(value_np.reshape(-1)[0])


def _calc_reprojection_errors(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec_m2c: np.ndarray,
    t_m2c: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Calculates per-correspondence reprojection errors in pixels."""

    projected_points, _ = cv2.projectPoints(
        object_points,
        rvec_m2c,
        t_m2c,
        K,
        None,
    )
    projected_points = projected_points.reshape(-1, 2)
    return np.linalg.norm(projected_points - image_points, axis=1)


def _calc_spatial_coverage(
    image_points: np.ndarray,
    inlier_ids: np.ndarray,
) -> float:
    """Scores how broadly inliers cover the correspondence footprint."""

    if len(inlier_ids) < 3 or len(image_points) < 3:
        return 0.0

    all_min = np.min(image_points, axis=0)
    all_max = np.max(image_points, axis=0)
    all_extent = np.maximum(all_max - all_min, 0.0)
    all_area = float(all_extent[0] * all_extent[1])
    if all_area <= 1e-6:
        return 0.0

    inlier_points = image_points[inlier_ids]
    inlier_min = np.min(inlier_points, axis=0)
    inlier_max = np.max(inlier_points, axis=0)
    inlier_extent = np.maximum(inlier_max - inlier_min, 0.0)
    inlier_area = float(inlier_extent[0] * inlier_extent[1])

    return float(np.clip(inlier_area / all_area, 0.0, 1.0))


def _calc_pose_quality(
    corresp: Dict[str, Any],
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec_m2c: np.ndarray,
    t_m2c: np.ndarray,
    inliers: Any,
    K: np.ndarray,
    pnp_inlier_thresh: float,
) -> Tuple[float, Dict[str, float]]:
    """Combines inlier, reprojection, template, and spatial metrics."""

    num_corresp = len(image_points)
    inlier_ids = np.asarray(
        [] if inliers is None else inliers, dtype=np.int64
    ).reshape(-1)
    inlier_count = int(len(inlier_ids))
    inlier_ratio = float(inlier_count / num_corresp) if num_corresp > 0 else 0.0

    errors = _calc_reprojection_errors(
        object_points=object_points,
        image_points=image_points,
        rvec_m2c=rvec_m2c,
        t_m2c=t_m2c,
        K=K,
    )
    inlier_errors = errors[inlier_ids] if inlier_count > 0 else np.asarray([])

    fallback_error = 2.0 * max(float(pnp_inlier_thresh), 1e-6)
    mean_reproj_error = (
        float(np.mean(inlier_errors)) if len(inlier_errors) > 0 else fallback_error
    )
    median_reproj_error = (
        float(np.median(inlier_errors)) if len(inlier_errors) > 0 else fallback_error
    )

    template_score = _value_to_float(corresp.get("template_score"), default=0.0)
    normalized_template_score = float(np.clip(template_score, 0.0, 1.0))
    spatial_coverage = _calc_spatial_coverage(
        image_points=image_points,
        inlier_ids=inlier_ids,
    )

    clipped_mean_reproj_error = float(
        min(mean_reproj_error / max(float(pnp_inlier_thresh), 1e-6), 2.0)
    )
    quality = (
        2.0 * inlier_ratio
        + normalized_template_score
        + spatial_coverage
        - 1.5 * clipped_mean_reproj_error
    )

    quality_info = {
        "inlier_count": float(inlier_count),
        "num_corresp": float(num_corresp),
        "inlier_ratio": inlier_ratio,
        "mean_reproj_error": mean_reproj_error,
        "median_reproj_error": median_reproj_error,
        "template_score": template_score,
        "normalized_template_score": normalized_template_score,
        "spatial_coverage": spatial_coverage,
        "clipped_mean_reproj_error": clipped_mean_reproj_error,
    }

    return float(quality), quality_info


def estimate_pose(
    corresp: Dict[str, Any],
    camera_c2w: PinholePlaneCameraModel,
    pnp_type: str,
    pnp_ransac_iter: int,
    pnp_inlier_thresh: float,
    pnp_required_ransac_conf: float,
    pnp_refine_lm: bool,
) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray, float, Dict[str, float]]:
    """Estimates pose from provided 2D-3D correspondences and camera intrinsics.

    Args:
        corresp: correspondence dictionary as returned by corresp_util. Has the following:
            - coord_2d (num_points, 2): pixel coordinates from query image
            - coord_3d (num_points, 3): point coordinates from the 3d object representation
            - nn_distances (num_points) : cosine distances as returned by KNN
            - nn_indices (num_points).: indices within the object representations
        camera_c2w: camera intrinsics.
    """

    quality_info = {
        "inlier_count": 0.0,
        "num_corresp": 0.0,
        "inlier_ratio": 0.0,
        "mean_reproj_error": 0.0,
        "median_reproj_error": 0.0,
        "template_score": _value_to_float(corresp.get("template_score"), default=0.0),
        "normalized_template_score": float(
            np.clip(
                _value_to_float(corresp.get("template_score"), default=0.0),
                0.0,
                1.0,
            )
        ),
        "spatial_coverage": 0.0,
        "clipped_mean_reproj_error": 0.0,
    }

    if pnp_type == "opencv":

        object_points = tensor_to_array(corresp["coord_3d"]).astype(np.float32)
        image_points = tensor_to_array(corresp["coord_2d"]).astype(np.float32)
        K = misc.get_intrinsic_matrix(camera_c2w)
        quality_info["num_corresp"] = float(len(image_points))
        try:
            pose_est_success, rvec_est_m2c, t_est_m2c, inliers = cv2.solvePnPRansac(
                objectPoints=object_points,
                imagePoints=image_points,
                cameraMatrix=K,
                distCoeffs=None,
                iterationsCount=pnp_ransac_iter,
                reprojectionError=pnp_inlier_thresh,
                confidence=pnp_required_ransac_conf,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except Exception:
            # Added to avoid a crash in cv2.solvePnPRansac due to too less correspondences
            # (even though more than 6 are provided, some of them may be colinear...).
            pose_est_success = False
            r_est_m2c = None
            t_est_m2c = None
            inliers = None
            quality = 0.0
        else:
            inlier_ids = np.asarray(
                [] if inliers is None else inliers, dtype=np.int64
            ).reshape(-1)

            # Optional LM refinement on inliers.
            if pose_est_success and pnp_refine_lm and len(inlier_ids) > 0:
                rvec_est_m2c, t_est_m2c = cv2.solvePnPRefineLM(
                    objectPoints=object_points[inlier_ids],
                    imagePoints=image_points[inlier_ids],
                    cameraMatrix=K,
                    distCoeffs=None,
                    rvec=rvec_est_m2c,
                    tvec=t_est_m2c,
            )

            quality = 0.0
            r_est_m2c = None
            if pose_est_success:
                r_est_m2c = cv2.Rodrigues(rvec_est_m2c)[0]
                quality, quality_info = _calc_pose_quality(
                    corresp=corresp,
                    object_points=object_points,
                    image_points=image_points,
                    rvec_m2c=rvec_est_m2c,
                    t_m2c=t_est_m2c,
                    inliers=inlier_ids,
                    K=K,
                    pnp_inlier_thresh=pnp_inlier_thresh,
                )
                inliers = inlier_ids

    else:
        raise ValueError(f"Unsupported PnP type: {pnp_type}")

    return pose_est_success, r_est_m2c, t_est_m2c, inliers, quality, quality_info
