# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

"""Synthesizes object templates using Blender for raw rendering and FoundPose for geometry-aware crop/metadata."""


from typing import Dict, List, NamedTuple, Optional, Tuple

import json
import os
import subprocess

import cv2
import numpy as np

from bop_toolkit_lib import dataset_params, inout
import bop_toolkit_lib.config as bop_config

from utils import config_util, geometry, json_util, logging, misc
from utils import misc as foundpose_misc
from utils import structs
from utils.misc import warp_depth_image, warp_image
from utils.renderer_base import RenderType
from utils.structs import AlignedBox2f, PinholePlaneCameraModel


class GenTemplatesOpts(NamedTuple):
    """Options that can be specified via the command line."""

    version: str
    object_dataset: str
    object_lids: Optional[List[int]] = None

    # Viewpoint options.
    num_viewspheres: int = 1
    min_num_viewpoints: int = 57
    num_inplane_rotations: int = 14
    images_per_view: int = 1

    # Mesh pre-processing options.
    max_num_triangles: int = 20000
    back_face_culling: bool = False
    texture_size: Tuple[int, int] = (1024, 1024)

    # Rendering options.
    ssaa_factor: float = 4.0
    background_type: str = "black"
    light_type: str = "multi_directional"

    # Cropping options.
    crop: bool = True
    crop_rel_pad: float = 0.2
    crop_size: Tuple[int, int] = (420, 420)

    # Other options.
    features_patch_size: int = 14
    save_templates: bool = True
    overwrite: bool = True
    debug: bool = True

    # Blender bridge options.
    blender_exec: str = "blender"
    blender_script: str = "render_templates_blender.py"
    blender_scene: str = ""
    blender_engine: str = "CYCLES"
    blender_samples: int = 64


def _save_jobs(jobs: List[Dict], jobs_path: str) -> None:
    with open(jobs_path, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2)



def _run_blender(opts: GenTemplatesOpts, jobs_path: str) -> None:
    cmd = [opts.blender_exec]
    if opts.blender_scene:
        cmd.extend(["-b", opts.blender_scene])
        reset_scene = "0"
    else:
        cmd.append("-b")
        reset_scene = "1"

    cmd.extend(
        [
            "-P",
            opts.blender_script,
            "--",
            "--jobs",
            jobs_path,
            "--engine",
            opts.blender_engine,
            "--samples",
            str(opts.blender_samples),
            "--reset-scene",
            reset_scene,
        ]
    )
    subprocess.run(cmd, check=True)



def _load_raw_render(job: Dict) -> Dict:
    rgba = cv2.imread(job["raw_rgb_path"], cv2.IMREAD_UNCHANGED)
    if rgba is None:
        raise FileNotFoundError(f"Failed to load Blender RGB output: {job['raw_rgb_path']}")

    if rgba.ndim != 3 or rgba.shape[2] not in (3, 4):
        raise ValueError(f"Unexpected Blender RGB shape {rgba.shape} for {job['raw_rgb_path']}")

    if rgba.shape[2] == 4:
        alpha = rgba[:, :, 3]
        bgr = rgba[:, :, :3]
    else:
        alpha = None
        bgr = rgba

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    depth = np.load(job["raw_depth_path"]).astype(np.float32)

    if alpha is not None:
        mask = np.where(alpha > 0, 255, 0).astype(np.uint8)
        raw_mask_path = job.get("raw_mask_path")
        if raw_mask_path and not os.path.exists(raw_mask_path):
            os.makedirs(os.path.dirname(raw_mask_path), exist_ok=True)
            inout.save_im(raw_mask_path, mask)
    else:
        raw_mask_path = job.get("raw_mask_path")
        if raw_mask_path and os.path.exists(raw_mask_path):
            mask = cv2.imread(raw_mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            raise ValueError(
                "No alpha channel or raw mask was found. Blender should render RGBA or provide raw_mask_path."
            )

    return {
        RenderType.COLOR: rgb,
        RenderType.DEPTH: depth,
        RenderType.MASK: mask,
    }



def synthesize_templates(opts: GenTemplatesOpts) -> None:
    datasets_path = bop_config.datasets_path

    np.random.seed(0)

    logger = logging.get_logger(level=logging.INFO if opts.debug else logging.WARNING)
    timer = misc.Timer(enabled=opts.debug)
    timer.start()

    object_lids = opts.object_lids
    bop_model_props = dataset_params.get_model_params(
        datasets_path=datasets_path, dataset_name=opts.object_dataset
    )
    if object_lids is None:
        object_lids = bop_model_props["obj_ids"]

    bop_test_split_props = dataset_params.get_split_params(
        datasets_path=datasets_path,
        dataset_name=opts.object_dataset,
        split="test",
    )

    bop_camera = dataset_params.get_camera_params(
        datasets_path=datasets_path, dataset_name=opts.object_dataset
    )

    logger.info("BOP camera details were loaded.")
    logger.info(f"Object lids: {object_lids}")

    bop_camera_width = bop_camera["im_size"][0]
    bop_camera_height = bop_camera["im_size"][1]
    max_image_side = max(bop_camera_width, bop_camera_height)
    image_side = opts.features_patch_size * int(max_image_side / opts.features_patch_size)
    camera_model = PinholePlaneCameraModel(
        width=image_side,
        height=image_side,
        f=(bop_camera["K"][0, 0], bop_camera["K"][1, 1]),
        c=(
            bop_camera["K"][0, 2] - 0.5 * (bop_camera_width - image_side),
            bop_camera["K"][1, 2] - 0.5 * (bop_camera_height - image_side),
        ),
    )
    render_camera_model = PinholePlaneCameraModel(
        width=int(camera_model.width * opts.ssaa_factor),
        height=int(camera_model.height * opts.ssaa_factor),
        f=(
            camera_model.f[0] * opts.ssaa_factor,
            camera_model.f[1] * opts.ssaa_factor,
        ),
        c=(
            camera_model.c[0] * opts.ssaa_factor,
            camera_model.c[1] * opts.ssaa_factor,
        ),
    )

    depth_range = bop_test_split_props["depth_range"]
    min_depth = 400.0
    max_depth = 800.0
    depth_range_size = max_depth - min_depth
    depth_cell_size = depth_range_size / float(opts.num_viewspheres)
    viewsphere_radii: List[float] = []
    for depth_cell_id in range(opts.num_viewspheres):
        viewsphere_radii.append(min_depth + (depth_cell_id + 0.5) * depth_cell_size)

    views_sphere = []
    for radius in viewsphere_radii:
        views_sphere += foundpose_misc.sample_views(
            min_n_views=opts.min_num_viewpoints,
            radius=radius,
            mode="fibonacci",
        )[0]
    logger.info(f"Sampled points on the sphere: {len(views_sphere)}")

    if opts.num_inplane_rotations == 1:
        views = views_sphere
    else:
        inplane_angle = 2 * np.pi / opts.num_inplane_rotations
        views = []
        for view_sphere in views_sphere:
            for inplane_id in range(opts.num_inplane_rotations):
                R_inplane = geometry.rotation_matrix_numpy(
                    inplane_angle * inplane_id, np.array([0, 0, 1])
                )[:3, :3]
                views.append(
                    {
                        "R": R_inplane.dot(view_sphere["R"]),
                        "t": R_inplane.dot(view_sphere["t"]),
                    }
                )
    logger.info(f"Number of views: {len(views)}")
    timer.elapsed("Time for setting up the stage")

    for object_lid in object_lids:
        logging.log_heading(logger, f"Object {object_lid} from {opts.object_dataset}")
        timer.start()

        dataset_torch_relpath = os.path.join(
            "templates", opts.version, opts.object_dataset, str(object_lid)
        )
        output_dir = os.path.join(bop_config.output_path, dataset_torch_relpath)
        if os.path.exists(output_dir) and not opts.overwrite:
            raise ValueError(f"Output directory already exists: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output will be saved to: {output_dir}")

        config_path = os.path.join(output_dir, "config.json")
        json_util.save_json(config_path, opts)

        templates_rgb_dir = os.path.join(output_dir, "rgb")
        templates_depth_dir = os.path.join(output_dir, "depth")
        templates_mask_dir = os.path.join(output_dir, "mask")
        raw_rgb_dir = os.path.join(output_dir, "raw_rgb")
        raw_depth_dir = os.path.join(output_dir, "raw_depth")
        raw_mask_dir = os.path.join(output_dir, "raw_mask")
        if opts.save_templates:
            os.makedirs(templates_rgb_dir, exist_ok=True)
            os.makedirs(templates_depth_dir, exist_ok=True)
            os.makedirs(templates_mask_dir, exist_ok=True)
        os.makedirs(raw_rgb_dir, exist_ok=True)
        os.makedirs(raw_depth_dir, exist_ok=True)
        os.makedirs(raw_mask_dir, exist_ok=True)

        model_path = bop_model_props["model_tpath"].format(obj_id=object_lid)
        metadata_list = []
        jobs: List[Dict] = []

        timer.elapsed("Time for preparing object data")

        template_counter = 0
        for view_id, view in enumerate(views):
            logger.info(
                f"Preparing Blender job for object {object_lid}, view {view_id}/{len(views)}..."
            )
            for _ in range(opts.images_per_view):
                trans_m2c = structs.RigidTransform(R=view["R"], t=view["t"])
                R_c2m = trans_m2c.R.T
                trans_c2m = structs.RigidTransform(R=R_c2m, t=-R_c2m.dot(trans_m2c.t))
                trans_c2m_matrix = misc.get_rigid_matrix(trans_c2m)

                jobs.append(
                    {
                        "template_id": template_counter,
                        "obj_id": object_lid,
                        "model_path": model_path,
                        "width": render_camera_model.width,
                        "height": render_camera_model.height,
                        "fx": render_camera_model.f[0],
                        "fy": render_camera_model.f[1],
                        "cx": render_camera_model.c[0],
                        "cy": render_camera_model.c[1],
                        "T_world_from_eye_cv": trans_c2m_matrix.tolist(),
                        "raw_rgb_path": os.path.join(
                            raw_rgb_dir, f"template_{template_counter:04d}.png"
                        ),
                        "raw_depth_path": os.path.join(
                            raw_depth_dir, f"template_{template_counter:04d}.npy"
                        ),
                        "raw_mask_path": os.path.join(
                            raw_mask_dir, f"template_{template_counter:04d}.png"
                        ),
                    }
                )
                template_counter += 1

        jobs_path = os.path.join(output_dir, "blender_jobs.json")
        _save_jobs(jobs, jobs_path)
        _run_blender(opts, jobs_path)

        template_counter = 0
        for job in jobs:
            logger.info(
                f"Post-processing Blender render for object {object_lid}, template {job['template_id']}..."
            )
            timer.start()

            render_camera_model_c2w = PinholePlaneCameraModel(
                width=job["width"],
                height=job["height"],
                f=(job["fx"], job["fy"]),
                c=(job["cx"], job["cy"]),
                T_world_from_eye=np.asarray(job["T_world_from_eye_cv"], dtype=np.float32),
            )
            output = _load_raw_render(job)

            ys, xs = output[RenderType.MASK].nonzero()
            box = np.array(foundpose_misc.calc_2d_box(xs, ys))
            object_box = AlignedBox2f(
                left=box[0],
                top=box[1],
                right=box[2],
                bottom=box[3],
            )

            if (
                object_box.left == 0
                or object_box.top == 0
                or object_box.right == render_camera_model_c2w.width - 1
                or object_box.bottom == render_camera_model_c2w.height - 1
            ):
                raise ValueError("The model does not fit the viewport.")

            if opts.crop:
                crop_box = foundpose_misc.calc_crop_box(box=object_box, make_square=True)
                crop_camera_model_c2w = foundpose_misc.construct_crop_camera(
                    box=crop_box,
                    camera_model_c2w=render_camera_model_c2w,
                    viewport_size=(
                        int(opts.crop_size[0] * opts.ssaa_factor),
                        int(opts.crop_size[1] * opts.ssaa_factor),
                    ),
                    viewport_rel_pad=opts.crop_rel_pad,
                )

                for output_key in list(output.keys()):
                    if output_key == RenderType.DEPTH:
                        output[output_key] = warp_depth_image(
                            src_camera=render_camera_model_c2w,
                            dst_camera=crop_camera_model_c2w,
                            src_depth_image=output[output_key],
                        )
                    elif output_key == RenderType.COLOR:
                        interpolation = (
                            cv2.INTER_AREA
                            if crop_box.width >= crop_camera_model_c2w.width
                            else cv2.INTER_LINEAR
                        )
                        output[output_key] = warp_image(
                            src_camera=render_camera_model_c2w,
                            dst_camera=crop_camera_model_c2w,
                            src_image=output[output_key],
                            interpolation=interpolation,
                        )
                    else:
                        output[output_key] = warp_image(
                            src_camera=render_camera_model_c2w,
                            dst_camera=crop_camera_model_c2w,
                            src_image=output[output_key],
                            interpolation=cv2.INTER_NEAREST,
                        )

                camera_model_c2w = crop_camera_model_c2w.copy()
                scale_factor = opts.crop_size[0] / float(crop_camera_model_c2w.width)
                camera_model_c2w.width = opts.crop_size[0]
                camera_model_c2w.height = opts.crop_size[1]
                camera_model_c2w.c = (
                    camera_model_c2w.c[0] * scale_factor,
                    camera_model_c2w.c[1] * scale_factor,
                )
                camera_model_c2w.f = (
                    camera_model_c2w.f[0] * scale_factor,
                    camera_model_c2w.f[1] * scale_factor,
                )
            else:
                camera_model_c2w = PinholePlaneCameraModel(
                    width=camera_model.width,
                    height=camera_model.height,
                    f=camera_model.f,
                    c=camera_model.c,
                    T_world_from_eye=np.asarray(job["T_world_from_eye_cv"], dtype=np.float32),
                )

            if opts.ssaa_factor != 1.0:
                target_size = (camera_model_c2w.width, camera_model_c2w.height)
                for output_key in list(output.keys()):
                    interpolation = cv2.INTER_AREA if output_key == RenderType.COLOR else cv2.INTER_NEAREST
                    output[output_key] = misc.resize_image(
                        image=output[output_key],
                        size=target_size,
                        interpolation=interpolation,
                    )

            trans_m2w = structs.RigidTransform(R=np.eye(3), t=np.zeros((3, 1)))
            visibility = 1.0

            ys, xs = output[RenderType.MASK].nonzero()
            box = np.array(foundpose_misc.calc_2d_box(xs, ys))
            object_box = AlignedBox2f(
                left=box[0],
                top=box[1],
                right=box[2],
                bottom=box[3],
            )

            rgb_image = np.asarray(np.clip(255.0 * output[RenderType.COLOR], 0, 255), np.uint8)
            depth_image = output[RenderType.DEPTH]

            timer.elapsed("Time for template generation")
            timer.start()

            rgb_path = os.path.join(templates_rgb_dir, f"template_{template_counter:04d}.png")
            logger.info(f"Saving template RGB {template_counter} to: {rgb_path}")
            inout.save_im(rgb_path, rgb_image)

            depth_path = os.path.join(templates_depth_dir, f"template_{template_counter:04d}.png")
            logger.info(f"Saving template depth map {template_counter} to: {depth_path}")
            inout.save_depth(depth_path, depth_image)

            mask_path = os.path.join(templates_mask_dir, f"template_{template_counter:04d}.png")
            logger.info(f"Saving template binary mask {template_counter} to: {mask_path}")
            inout.save_im(mask_path, output[RenderType.MASK])

            data = {
                "dataset": opts.object_dataset,
                "lid": object_lid,
                "template_id": template_counter,
                "pose": trans_m2w,
                "boxes_amodal": np.array([object_box.array_ltrb()]).tolist(),
                "visibilities": np.array([visibility]).tolist(),
                "cameras": camera_model_c2w.to_json(),
                "rgb_image_path": rgb_path,
                "depth_map_path": depth_path,
                "binary_mask_path": mask_path,
            }
            timer.elapsed("Time for template saving")
            metadata_list.append(data)
            template_counter += 1

        metadata_path = os.path.join(output_dir, "metadata.json")
        json_util.save_json(metadata_path, metadata_list)


def main() -> None:
    opts = config_util.load_opts_from_json_or_command_line(GenTemplatesOpts)[0]
    synthesize_templates(opts)


if __name__ == "__main__":
    main()
