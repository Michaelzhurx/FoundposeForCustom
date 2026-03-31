import os, json, argparse
import numpy as np
import cv2
import trimesh
import pyrender
import matplotlib.pyplot as plt


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def find_file_any(base_dir, candidates):
    for rel in candidates:
        p = os.path.join(base_dir, rel)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("None of these files exist:\n" + "\n".join([os.path.join(base_dir, c) for c in candidates]))


def pose_from_entry(entry):
    """
    Support both:
    - BOP gt format: cam_R_m2c (len=9), cam_t_m2c (len=3)
    - FoundPose-like: R (len=9 or 3x3), t (len=3)
    Return 4x4 pose in camera coordinates.
    """
    if "cam_R_m2c" in entry:
        R = np.array(entry["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
        t = np.array(entry["cam_t_m2c"], dtype=np.float32).reshape(3, 1)
    elif "R" in entry and "t" in entry:
        R = np.array(entry["R"], dtype=np.float32).reshape(3, 3)
        t = np.array(entry["t"], dtype=np.float32).reshape(3, 1)
    else:
        raise KeyError(f"Unknown pose keys in entry: {entry.keys()}")

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3:4] = t
    return T


def render_silhouette(mesh_trimesh, K, pose_cam, W, H):
    """
    Render object only (white on black), return binary mask and rendered RGB.
    pose_cam: 4x4 object-to-camera transform.
    """
    # pyrender expects OpenGL camera looking down -Z in its convention; but IntrinsicsCamera in pyrender
    # uses same pinhole model; we can directly feed pose as given (object->camera) by placing mesh with that pose
    mesh = pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=False)

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.2, 0.2, 0.2])

    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=1e-3, zfar=1e6)
    scene.add(cam, pose=np.eye(4, dtype=np.float32))

    # Put mesh in camera coordinates using pose_cam
    scene.add(mesh, pose=pose_cam)

    # Add a light (in camera frame)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=np.eye(4, dtype=np.float32))

    r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    color, depth = r.render(scene)
    r.delete()

    # silhouette by thresholding (object pixels are non-black)
    gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
    mask = (gray > 5).astype(np.uint8) * 255
    return mask, color


def draw_contour(img_bgr, mask, color_bgr, thickness=2):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(cnts) == 0:
        return img_bgr
    # take largest
    cnt = max(cnts, key=cv2.contourArea)
    cv2.drawContours(img_bgr, [cnt], -1, color_bgr, thickness)
    return img_bgr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bop_path", type=str, required=True)
    ap.add_argument("--dataset", type=str, default="lmo")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--scene_id", type=int, required=True)
    ap.add_argument("--im_id", type=int, required=True)
    ap.add_argument("--obj_id", type=int, default=1)
    ap.add_argument("--method", type=str, default="lmo_v1")  # inference folder name
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    bop = args.bop_path
    ds = args.dataset
    split = args.split
    scene = f"{args.scene_id:06d}"
    im = f"{args.im_id:06d}"
    obj_id = args.obj_id

    base = os.path.join(bop, ds, split, scene)

    # 1) load image (try rgb first; fallback to other common names)
    img_path = find_file_any(base, [
        f"rgb/{im}.png",
        f"rgb/{im}.jpg",
        f"gray/{im}.png",
        f"gray/{im}.jpg",
        f"images/{im}.png",
        f"images/{im}.jpg",
    ])
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    H, W = img_bgr.shape[:2]

    # 2) intrinsics K
    cam_path = os.path.join(base, "scene_camera.json")
    cam_json = load_json(cam_path)
    cam_info = cam_json[str(args.im_id)]
    K = np.array(cam_info["cam_K"], dtype=np.float32).reshape(3, 3)

    # 3) load GT pose for obj_id (if exists)
    gt_path = os.path.join(base, "scene_gt.json")
    gt_json = load_json(gt_path)
    gt_list = gt_json[str(args.im_id)]
    gt_entry = None
    for e in gt_list:
        if int(e.get("obj_id", -1)) == obj_id:
            gt_entry = e
            break

    # 4) load estimated pose from FoundPose output
    est_path = os.path.join(bop, "inference", args.method, str(obj_id), "estimated-poses.json")
    est_json = load_json(est_path)

    # support either list or dict container
    if isinstance(est_json, dict) and "poses" in est_json:
        est_list = est_json["poses"]
    else:
        est_list = est_json

    est_entry = None
    for e in est_list:
        # try common keys
        sid = int(e.get("scene_id", -1))
        iid = int(e.get("im_id", -1))
        oid = int(e.get("obj_id", obj_id))
        if sid == args.scene_id and iid == args.im_id and oid == obj_id:
            est_entry = e
            break
    if est_entry is None:
        # fallback: first entry (if user generated per-image file)
        if len(est_list) > 0:
            est_entry = est_list[0]
        else:
            raise RuntimeError(f"No entries found in {est_path}")

    # Convert poses
    est_pose = pose_from_entry(est_entry)
    gt_pose = pose_from_entry(gt_entry) if gt_entry is not None else None

    # 5) load mesh (BOP models)
    # common BOP path: bop/dataset/models/obj_000001.ply OR models_eval/...
    model_dir = os.path.join(bop, ds, "models")
    model_path = None
    for cand_dir in [model_dir, os.path.join(bop, ds, "models_eval")]:
        cand = os.path.join(cand_dir, f"obj_{obj_id:06d}.ply")
        if os.path.exists(cand):
            model_path = cand
            break
    if model_path is None:
        raise FileNotFoundError(f"Cannot find model ply for obj {obj_id}. Tried under {model_dir} and models_eval.")

    mesh = trimesh.load(model_path, force="mesh")

    # 6) render silhouettes and overlay contours
    vis_bgr = img_bgr.copy()

    # GT in green
    if gt_pose is not None:
        gt_mask, _ = render_silhouette(mesh, K, gt_pose, W, H)
        vis_bgr = draw_contour(vis_bgr, gt_mask, (0, 255, 0), thickness=2)

    # Estimated in red
    est_mask, est_render = render_silhouette(mesh, K, est_pose, W, H)
    vis_bgr = draw_contour(vis_bgr, est_mask, (0, 0, 255), thickness=2)

    # 7) make a "similar to paper" layout: left = RGB+contours, right = rendered pose
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    est_render_rgb = est_render  # already RGB

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"{ds.upper()} {split} s{scene} i{im} obj{obj_id}\nGT (green) vs Est (red)")
    plt.imshow(vis_rgb)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Rendered (Est pose)")
    plt.imshow(est_render_rgb)
    plt.axis("off")

    out_dir = args.out_dir or os.path.join(bop, "vis", args.method)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ds}_{split}_s{scene}_i{im}_obj{obj_id}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print("Saved to:", out_path)
    plt.show()


if __name__ == "__main__":
    main()
