import os
import cv2

# Paths
image_dir = "/petneus/public_data/banc/images_4k"
mask_dir = "/petneus/public_data/banc/masks_4k"

out_image_dir = "/petneus/public_data/banc/images_1600"
out_mask_dir = "/petneus/public_data/banc/masks_1600"

# Ensure output folders exist
os.makedirs(out_image_dir, exist_ok=True)
os.makedirs(out_mask_dir, exist_ok=True)

# Target max resolution
max_res = 1600

def resize_image(img, is_mask=False):
    h, w = img.shape[:2]
    scale = max_res / max(h, w)
    if scale >= 1.0:
        return img  # already small enough
    new_w, new_h = int(w * scale), int(h * scale)
    if is_mask:
        # masks need nearest-neighbor to keep crisp binary edges
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

# Process images
for fname in sorted(os.listdir(image_dir)):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        img = cv2.imread(os.path.join(image_dir, fname), cv2.IMREAD_UNCHANGED)
        out = resize_image(img, is_mask=False)
        cv2.imwrite(os.path.join(out_image_dir, fname), out)

# Process masks
for fname in sorted(os.listdir(mask_dir)):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        mk = cv2.imread(os.path.join(mask_dir, fname), cv2.IMREAD_UNCHANGED)
        out = resize_image(mk, is_mask=True)
        cv2.imwrite(os.path.join(out_mask_dir, fname), out)

print("✅ Done! Resized images in:", out_image_dir, "and masks in:", out_mask_dir)
