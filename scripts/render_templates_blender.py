#!/usr/bin/env python3
"""Blender-side raw renderer for FoundPose template generation.

Usage:
    blender -b -P render_templates_blender.py -- \
        --jobs /path/to/blender_jobs.json \
        --engine CYCLES \
        --samples 64 \
        --reset-scene 1
"""


import argparse
import json
import os
import sys
from typing import Dict, List

import bpy
import numpy as np
from mathutils import Matrix


CV_TO_BLENDER = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)



def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--engine", default="CYCLES")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--reset-scene", type=int, default=1)
    return parser.parse_args(argv)



def clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)



def configure_scene(engine: str, samples: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.use_file_extension = True
    scene.render.resolution_percentage = 100

    view_layer = scene.view_layers[0]
    view_layer.use_pass_z = True

    if engine == "CYCLES":
        scene.cycles.samples = samples
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.device = "CPU"
    elif engine.startswith("BLENDER_EEVEE"):
        scene.eevee.taa_render_samples = samples

    scene.use_nodes = True
    nt = scene.node_tree
    nt.nodes.clear()

    render_layers = nt.nodes.new(type="CompositorNodeRLayers")

    depth_out = nt.nodes.new(type="CompositorNodeOutputFile")
    depth_out.label = "DepthOutput"
    depth_out.name = "DepthOutput"
    depth_out.base_path = ""
    depth_out.format.file_format = "OPEN_EXR"
    depth_out.format.color_depth = "32"
    depth_out.format.color_mode = "RGB"

    nt.links.new(render_layers.outputs["Depth"], depth_out.inputs[0])

    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
        bg.inputs[1].default_value = 0.0



def ensure_output_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)



def add_default_lights() -> None:
    scene = bpy.context.scene
    if any(obj.type == "LIGHT" for obj in scene.objects):
        return

    lights = [
        ("SUN", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 2.0),
        ("AREA", (0.35, -0.55, 0.65), (1.0, 0.2, 0.5), 3000.0),
        ("AREA", (-0.55, -0.35, 0.45), (1.2, 0.0, -0.7), 1500.0),
        ("AREA", (0.0, 0.65, -0.2), (1.3, 0.0, 0.0), 1000.0),
    ]

    for idx, (light_type, location, rotation, energy) in enumerate(lights):
        data = bpy.data.lights.new(name=f"TemplateLight{idx}", type=light_type)
        data.energy = energy
        if light_type == "AREA":
            data.shape = "RECTANGLE"
            data.size = 1.0
            data.size_y = 1.0
        obj = bpy.data.objects.new(name=f"TemplateLight{idx}", object_data=data)
        obj.location = location
        obj.rotation_euler = rotation
        bpy.context.scene.collection.objects.link(obj)



def import_model(model_path: str):
    ext = os.path.splitext(model_path)[1].lower()
    before = set(bpy.data.objects.keys())

    if ext == ".ply":
        try:
            bpy.ops.wm.ply_import(filepath=model_path)
        except Exception:
            bpy.ops.import_mesh.ply(filepath=model_path)
    elif ext == ".stl":
        try:
            bpy.ops.wm.stl_import(filepath=model_path)
        except Exception:
            bpy.ops.import_mesh.stl(filepath=model_path)
    elif ext == ".obj":
        try:
            bpy.ops.wm.obj_import(filepath=model_path)
        except Exception:
            bpy.ops.import_scene.obj(filepath=model_path)
    else:
        raise ValueError(f"Unsupported model format: {ext}")

    after = set(bpy.data.objects.keys())
    new_names = list(after - before)
    new_objects = [bpy.data.objects[name] for name in new_names]
    mesh_objects = [obj for obj in new_objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError(f"No mesh objects were imported from: {model_path}")

    root_name = f"ImportedModel::{os.path.basename(model_path)}"
    root = bpy.data.objects.new(root_name, None)
    bpy.context.scene.collection.objects.link(root)
    root.matrix_world = Matrix.Identity(4)

    for obj in mesh_objects:
        obj.parent = root

    return root, mesh_objects



def ensure_materials(mesh_objects) -> None:
    for obj in mesh_objects:
        if obj.data.materials:
            continue
        mat = bpy.data.materials.new(name=f"Mat::{obj.name}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = (0.7, 0.7, 0.7, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.4
            bsdf.inputs["Metallic"].default_value = 0.0
        obj.data.materials.append(mat)



def create_camera() -> bpy.types.Object:
    cam_data = bpy.data.cameras.new("TemplateCamera")
    cam_obj = bpy.data.objects.new("TemplateCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    cam_data.clip_start = 0.01
    cam_data.clip_end = 1e6
    return cam_obj



def set_camera_intrinsics(cam_obj, width: int, height: int, fx: float, fy: float, cx: float, cy: float) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)

    cam = cam_obj.data
    cam.sensor_fit = "HORIZONTAL"
    sensor_width = cam.sensor_width
    cam.lens = float(fx) * sensor_width / float(width)
    cam.shift_x = (float(cx) - float(width) * 0.5) / float(width)
    cam.shift_y = (float(height) * 0.5 - float(cy)) / float(width)



def set_camera_pose(cam_obj, T_world_from_eye_cv: np.ndarray) -> None:
    T_world_from_eye_blender = T_world_from_eye_cv @ CV_TO_BLENDER
    cam_obj.matrix_world = Matrix(T_world_from_eye_blender.tolist())



def render_rgb_and_depth(rgb_path: str, depth_path: str) -> None:
    scene = bpy.context.scene
    ensure_output_dir(rgb_path)
    ensure_output_dir(depth_path)

    scene.render.filepath = rgb_path

    depth_exr_prefix = os.path.splitext(depth_path)[0]
    depth_node = scene.node_tree.nodes["DepthOutput"]
    depth_node.base_path = os.path.dirname(depth_exr_prefix)
    depth_node.file_slots[0].path = os.path.basename(depth_exr_prefix)

    bpy.ops.render.render(write_still=True)

    depth_dir = os.path.dirname(depth_exr_prefix)
    depth_base = os.path.basename(depth_exr_prefix)

    candidates = [
        os.path.join(depth_dir, f)
        for f in os.listdir(depth_dir)
        if f.startswith(depth_base) and f.endswith(".exr")
    ]
    if not candidates:
        raise RuntimeError(f"No EXR depth file found for prefix: {depth_exr_prefix}")

    depth_exr_path = sorted(candidates)[-1]

    # 用 Blender 自己读取 EXR，不依赖 imageio
    img = bpy.data.images.load(depth_exr_path, check_existing=False)
    try:
        width, height = img.size
        channels = img.channels
        pixels = np.array(img.pixels[:], dtype=np.float32)

        if pixels.size != width * height * channels:
            raise RuntimeError(
                f"Unexpected EXR pixel buffer size: got {pixels.size}, "
                f"expected {width * height * channels}."
            )

        depth_arr = pixels.reshape((height, width, channels))

        # 深度取第一通道
        depth = depth_arr[:, :, 0]

        # 和前面流程保持一致
        depth = np.flipud(depth)

        invalid = ~np.isfinite(depth)
        depth[invalid] = 0.0

        # 转毫米
        depth_mm = depth * 1000.0
        np.save(depth_path, depth_mm)
    finally:
        bpy.data.images.remove(img)

    # 可选：删掉中间 EXR
    # os.remove(depth_exr_path)



def main() -> None:
    args = parse_args()
    with open(args.jobs, "r", encoding="utf-8") as f:
        jobs: List[Dict] = json.load(f)

    if not jobs:
        raise ValueError("No jobs were found in the provided jobs file.")

    if args.reset_scene:
        clear_scene()

    configure_scene(args.engine, args.samples)
    add_default_lights()

    model_paths = sorted({job["model_path"] for job in jobs})
    if len(model_paths) != 1:
        raise ValueError(
            "This script expects all jobs in one run to share the same model path. "
            f"Got: {model_paths}"
        )

    root, mesh_objects = import_model(model_paths[0])
    ensure_materials(mesh_objects)
    cam_obj = create_camera()

    for job in jobs:
        set_camera_intrinsics(
            cam_obj,
            width=int(job["width"]),
            height=int(job["height"]),
            fx=float(job["fx"]),
            fy=float(job["fy"]),
            cx=float(job["cx"]),
            cy=float(job["cy"]),
        )
        set_camera_pose(
            cam_obj,
            np.asarray(job["T_world_from_eye_cv"], dtype=np.float32),
        )
        render_rgb_and_depth(job["raw_rgb_path"], job["raw_depth_path"])


if __name__ == "__main__":
    main()
