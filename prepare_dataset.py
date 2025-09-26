#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import cv2


# ---------- Utilities ----------
def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ---------- COLMAP ----------
def run_colmap_reconstruction(images_dir: Path, work_dir: Path, colmap_bin: str,
                              matcher: str, camera_model: str) -> Tuple[Path, Path]:
    db_path = work_dir / "colmap.db"
    sparse_dir = work_dir / "sparse"
    txt_dir = work_dir / "colmap_text"
    ply_dir = work_dir / "colmap_ply"
    ply_file = ply_dir / "points3D.ply"

    # ensure dirs
    ensure_dir(work_dir)
    ensure_dir(sparse_dir)
    ensure_dir(txt_dir)
    ensure_dir(ply_dir)

    run([colmap_bin, "feature_extractor",
         "--database_path", str(db_path),
         "--image_path", str(images_dir),
         "--ImageReader.camera_model", camera_model,
         "--ImageReader.single_camera", "1"])
    run([colmap_bin, f"{matcher}_matcher" if matcher != "exhaustive" else "exhaustive_matcher",
         "--database_path", str(db_path)])
    run([colmap_bin, "mapper",
         "--database_path", str(db_path),
         "--image_path", str(images_dir),
         "--output_path", str(sparse_dir)])
    run([colmap_bin, "model_converter",
         "--input_path", str(sparse_dir / "0"),
         "--output_path", str(txt_dir),
         "--output_type", "TXT"])
    run([colmap_bin, "model_converter",
         "--input_path", str(sparse_dir / "0"),
         "--output_path", str(ply_file),
         "--output_type", "PLY"])
    return txt_dir, ply_file


def parse_colmap_cameras(cameras_txt: Path) -> Tuple[np.ndarray, int, int]:
    with open(cameras_txt, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('#') or not line.strip():
            continue
        tokens = line.split()
        model, width, height = tokens[1], int(tokens[2]), int(tokens[3])
        params = list(map(float, tokens[4:]))
        if model in ['SIMPLE_PINHOLE', 'PINHOLE']:
            if model == 'SIMPLE_PINHOLE':
                f = params[0]; cx = params[1]; cy = params[2]
                K = np.array([[f,0,cx],[0,f,cy],[0,0,1]])
            else:
                fx, fy, cx, cy = params[:4]
                K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
        else:
            fx, fy, cx, cy = params[:4]
            K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
        return K, width, height
    raise RuntimeError("No valid camera found in cameras.txt")


def qvec2rotmat(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1-2*(qx*qx+qy*qy)]
    ])


def parse_colmap_images(images_txt: Path) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    data = []
    with open(images_txt, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip(); i += 1
        if line.startswith('#') or not line: continue
        toks = line.split()
        qw,qx,qy,qz = map(float, toks[1:5])
        tx,ty,tz = map(float, toks[5:8])
        name = toks[9]; i += 1
        R_c2w = qvec2rotmat(np.array([qw,qx,qy,qz]))
        t_c2w = np.array([tx,ty,tz])
        data.append((name,R_c2w,t_c2w))
    return sorted(data, key=lambda x:x[0])


def build_camera_mats(K: np.ndarray, poses: List[Tuple[str,np.ndarray,np.ndarray]]) -> Dict[str,np.ndarray]:
    cams = {}
    for i,(_,R_c2w,t_c2w) in enumerate(poses):
        R = R_c2w.T
        t = -R @ t_c2w
        P = np.eye(4)
        P[:3,:3] = K @ R
        P[:3,3]  = K @ t
        cams[f"world_mat_{i}"] = P.astype(np.float32)
    return cams


# ---------- PLY helpers ----------
def load_ply_xyz(ply_path: Path) -> np.ndarray:
    try:
        import trimesh
        return np.asarray(trimesh.load(str(ply_path), process=False).vertices, np.float32)
    except Exception:
        xyz=[]; header=True; count=0
        with open(ply_path,'r') as f:
            for ln in f:
                if header:
                    if ln.startswith("element vertex"): count=int(ln.split()[-1])
                    if ln.strip()=="end_header": header=False; continue
                else:
                    parts=ln.split()
                    if len(parts)>=3: xyz.append([float(parts[0]),float(parts[1]),float(parts[2])])
                    if len(xyz)>=count: break
        return np.asarray(xyz,np.float32)


def compute_scale_mat_from_points(xyz: np.ndarray, target_radius=1.0) -> np.ndarray:
    c = np.median(xyz, axis=0)
    r = np.percentile(np.linalg.norm(xyz-c,axis=1), 99.0)+1e-6
    s = target_radius/r
    M = np.eye(4,dtype=np.float32)
    M[0,0]=M[1,1]=M[2,2]=s
    M[:3,3]=-s*c
    return M


# ---------- Images & Masks ----------
def copy_and_rename_images(src_dir: Path, dst_dir: Path):
    if dst_dir.exists(): shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True)
    for i,p in enumerate(sorted(src_dir.iterdir())):
        if p.suffix.lower() not in [".jpg",".jpeg",".png"]: continue
        img=cv2.imread(str(p)); cv2.imwrite(str(dst_dir/f"{i:06d}.png"), img)


def save_masks(images_dir: Path, masks_dir: Path):
    if masks_dir.exists(): shutil.rmtree(masks_dir)
    masks_dir.mkdir(parents=True)
    for i,p in enumerate(sorted(images_dir.iterdir())):
        if p.suffix.lower()!=".png": continue
        img=cv2.imread(str(p)); h,w=img.shape[:2]
        m=np.ones((h,w),np.uint8)*255
        cv2.imwrite(str(masks_dir/f"{i:06d}.png"), m)


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser("Prepare PET-NeuS dataset (with optional cleaned PLY)")
    parser.add_argument('--video', type=str, default=None)
    parser.add_argument('--images', type=str, default=None)
    parser.add_argument('--fps', type=float, default=2)
    parser.add_argument('--scene_name', type=str, required=True)
    parser.add_argument('--output_root', type=str, default='public_data')
    parser.add_argument('--colmap_bin', type=str, default='colmap')
    parser.add_argument('--colmap_matcher', type=str, default='sequential')
    parser.add_argument('--camera_model', type=str, default='OPENCV')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--cleaned_ply', type=str, default=None,
                        help="Path to cleaned PLY for normalization")
    args = parser.parse_args()

    scene_dir = Path(args.output_root)/args.scene_name
    images_tmp = Path(args.images) if args.images else scene_dir/'raw_images'
    work_dir = scene_dir/'colmap'

    # Extract frames if video
    if args.video:
        ensure_dir(work_dir/"raw_images")
        subprocess.run([
            "ffmpeg", "-i", args.video,
            "-vf", f"fps={args.fps}",
            str(work_dir/"raw_images/frame_%05d.png")
        ], check=True)

    elif not images_tmp.exists():
        raise RuntimeError(f"Images dir not found: {images_tmp}")

    # Run COLMAP + export TXT + PLY
    txt_dir, default_ply = run_colmap_reconstruction(images_tmp, work_dir, args.colmap_bin, args.colmap_matcher, args.camera_model)

    # Cameras & poses
    K,W,H = parse_colmap_cameras(txt_dir/'cameras.txt')
    poses = parse_colmap_images(txt_dir/'images.txt')
    cams = build_camera_mats(K, poses)

    # Normalization: prefer cleaned PLY if provided
    ply_for_norm = Path(args.cleaned_ply) if args.cleaned_ply else default_ply
    print(f"Using PLY for normalization: {ply_for_norm}")
    xyz = load_ply_xyz(ply_for_norm)
    scale_mat = compute_scale_mat_from_points(xyz)

    # Save npz
    npz_dict = {}
    npz_dict.update(cams)
    for i in range(len(poses)):
        npz_dict[f"scale_mat_{i}"]=scale_mat
    np.savez(scene_dir/'cameras_sphere.npz', **npz_dict)

    # Copy images & masks
    copy_and_rename_images(images_tmp, scene_dir/'image')
    save_masks(scene_dir/'image', scene_dir/'mask')

    print(f"Dataset ready at {scene_dir}")
    print(f"COLMAP raw PLY saved to {default_ply}")
    if args.cleaned_ply:
        print(f"Cleaned PLY was applied: {args.cleaned_ply}")


if __name__=="__main__":
    main()
