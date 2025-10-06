import numpy as np

def analyze_npz(path, label):
    npz = np.load(path)
    world_mats = [npz[k] for k in npz.keys() if k.startswith("world_mat_")]
    scale_mats = [npz[k] for k in npz.keys() if k.startswith("scale_mat_")]

    # Extract intrinsics from first world_mat (approx, since it’s K @ [R|t])
    W0 = world_mats[0]
    K_approx = W0[:3, :3]  # not pure K but gives an idea of scale

    scales = [S[0,0] for S in scale_mats]  # since we set isotropic scale
    print(f"\n=== {label} ===")
    print("Num cams:", len(world_mats))
    print("Scale mean/min/max:", np.mean(scales), np.min(scales), np.max(scales))
    print("First world_mat:\n", world_mats[0])
    print("First scale_mat:\n", scale_mats[0])

    return np.array(scales), world_mats

# Compare yours vs scan106
scales_banc, world_banc = analyze_npz("public_data/banc/cameras_sphere.npz", "BANC")
scales_scan, world_scan = analyze_npz("public_data/scan106/cameras_sphere.npz", "SCAN106")
