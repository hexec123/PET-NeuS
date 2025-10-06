import numpy as np
from pathlib import Path

npz_path = Path("public_data/banc/cameras_sphere.npz")
data = dict(np.load(npz_path))

# Extract camera centers from projection matrices
centers = []
for k in data.keys():
    if k.startswith("world_mat_"):
        P = data[k][:3, :4]
        K, Rt = P[:, :3], P
        # Quick pseudo-inverse for center (approx)
        C = -np.linalg.pinv(Rt[:, :3]) @ Rt[:, 3]
        centers.append(C)
centers = np.array(centers)

# Compute bounding box
mins = centers.min(axis=0)
maxs = centers.max(axis=0)
center = (mins + maxs) / 2
extent = (maxs - mins).max()

print("Camera center bbox:")
print("min:", mins, "max:", maxs)
print("extent:", extent)

# Build normalization transform
scale = 2.0 / extent   # fit into [-1,1]
translate = -center

# Update all scale_mats
for k in data.keys():
    if k.startswith("scale_mat_"):
        S = np.eye(4, dtype=np.float32)
        S[:3, :3] *= scale
        S[:3, 3] = scale * translate
        data[k] = S

# Save back
np.savez(npz_path, **data)
print(f"Rewritten {npz_path} with normalized scale/translation.")
