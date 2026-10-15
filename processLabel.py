import cv2
import numpy as np
from skimage.morphology import dilation, disk, erosion
from skimage.io import imsave
import os
from PIL import Image
# -----------------------
# Configuration
# -----------------------
experiment = "CL"
borehole_name = "BCL11"

in_path  = f"data/Labeled Data Myriam/{experiment}/{borehole_name}_structure_new.tif"
# in_path = f"data/general_borehole_testing/annotation/BCL11.tif"
optv_path = f"data/Labeled Data Myriam/{experiment}/{borehole_name}_OPTV.tif"
out_path  = f"data/general_borehole_testing/annotation/{borehole_name}.png"
# out_path = f"data/general-boreholes-training/label_multiclass/{borehole_name}.png"

# Size of the OPTV image (for reference)
image_width,image_height = Image.open(optv_path).size
print(f"OPTV image size: width={image_width}, height={image_height}")

img =  Image.open(in_path)
boundary_pixel = 0
crop_rectangle = (boundary_pixel, boundary_pixel, img.size[0]-boundary_pixel, img.size[1]-boundary_pixel)
cropped_im = img.crop(crop_rectangle)
img_resized = cropped_im.resize((image_width,image_height), Image.BICUBIC)
# img_resized.save(f"data/Labeled Data Myriam/{experiment}/{borehole_name}_structures.png")

# Dilation kernel radius (adjust if needed)
DILATE_RADIUS = 10
selem = disk(DILATE_RADIUS)
EROSION_RADIUS = 2 
selem_erosion = disk(EROSION_RADIUS)

# -----------------------
# Load & convert
# -----------------------

img_np = np.array(img_resized) 

# 3. Convert RGB (Pillow) to BGR (OpenCV)
# OpenCV uses BGR, so if we don't convert, your Red and Blue channels will be swapped.
img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

# 4. Now run your original HSV line
# img_bgr is now a valid numpy array compatible with cv2
hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
H, W = hsv.shape[:2]

# -----------------------
# Define classes in PRIORITY order (top wins in conflicts).
# Use lists of HSV ranges for colors that need multiple bands (e.g., red wraps).
# Hue: 0-179, Sat/Val: 0-255
# -----------------------
# Tune these thresholds to your dataset if needed.
CLASSES = [
    # Specific, high-saturation colors FIRST
    {
        "name": "Intact wall (white)",
        "ranges": [((0, 0, 200), (179, 40, 255))]  # low S, high V
    },
    {
        "name": "Tectonic fault plane (red)",
        "ranges": [((0, 70, 50), (10, 255, 255)), ((170, 70, 50), (179, 255, 255))]
    },
    {
        "name": "Bedding (green)",
        "ranges": [((35, 60, 40), (95, 255, 255))]
    },
    {
        "name": "Fault zone (orange)",
        "ranges": [((10, 120, 100), (25, 255, 255))]
    },
    {
        "name": "EDZ (yellow)",
        "ranges": [((20, 100, 120), (35, 255, 255))]
    },
    # NEW: Medium grey (RGB 128,128,128)
    {
        "name": "Unclassified (gray)",
        "ranges": [((0, 0, 100), (179, 40, 160))]  # low S, medium V
    },
    
    # {
    #     "name": "Induced crack (light blue/cyan)",
    #     "ranges": [((85, 80, 70), (105, 255, 255))]
    # },

    # # Broad catch-alls LAST
    {
        "name": "Fault gouge (black / very dark)",
        "ranges": [((0, 0, 0), (179, 255, 50))]  # very low V
    },
    
]

# -----------------------
# Step 1: Build non-overlapping per-class masks
# -----------------------
assigned = np.zeros((H, W), dtype=bool)      # track pixels that already got a class
label_map = np.full((H, W), fill_value=255, dtype=np.uint8)  # 255 means "unassigned"
class_masks = []   # store clean, non-overlapping masks per class (before dilation)
label_names = []

for label_id, cls in enumerate(CLASSES):
    name = cls["name"]
    # OR all provided ranges for this class
    m = np.zeros((H, W), dtype=np.uint8)
    for lo, hi in cls["ranges"]:
        m = cv2.bitwise_or(m, cv2.inRange(hsv, lo, hi))

    # Remove pixels already assigned to earlier (higher-priority) classes
    m_clean = (m > 0) & (~assigned)

    # Save clean mask and update trackers
    class_masks.append(m_clean)
    label_names.append((label_id, name))
    label_map[m_clean] = label_id
    assigned |= m_clean

# -----------------------
# Step 2: Dilate each class independently
# -----------------------
MIN_AREA_THRESHOLD = 300  # minimum area to keep a component (in pixels)
dilated_masks = []
for label_id, m in enumerate(class_masks):
    if np.any(m):
        # --- 1. Filter by Area (Remove Dots) ---
        
        # Find all connected components (blobs)
        # connectivity=8 means diagonal pixels connect (important for curves)
        m_uint8 = m.astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(m_uint8, connectivity=8)
        
        # Create a new mask for the clean version
        clean_m = np.zeros_like(m_uint8)
        
        # Loop through all found components (start from 1 to skip background)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            # If the blob is big enough, keep it
            if area >= MIN_AREA_THRESHOLD:
                clean_m[labels == i] = 255
        
        # --- 2. Dilate the Cleaned Mask ---
        # Now we dilate the "dot-free" lines
        d = dilation(clean_m, selem)
        # d = erosion(d, selem_erosion)
        # d = dilation(m, selem)
    else:
        d = m
    dilated_masks.append(d.astype(bool))
    # Debug save dilated mask
    # cv2.imwrite(os.path.join(debug_dir, f"{label_id:02d}_{label_names[label_id][1].replace(' ', '_')}_dil.png"), (d.astype(np.uint8)*255))

# -----------------------
# Step 3: Merge dilated masks with priority (same as class order)
# Higher-priority classes overwrite lower-priority ones in overlaps.
# -----------------------
final_map = np.full((H, W), fill_value=255, dtype=np.uint8)  # keep 255 to spot holes
for label_id in range(len(dilated_masks)):
    final_map[dilated_masks[label_id]] = label_id

# Optionally, force any remaining 255 to Unclassified if you prefer a dense labeling:
# final_map[final_map == 255] = len(CLASSES) - 1

imsave(out_path, final_map.astype(np.uint8))

# -----------------------
# Diagnostics
# -----------------------
unique_vals, counts = np.unique(final_map, return_counts=True)
# print(f"Saved RAW label map: {out_mask_raw}")
print(f"Saved DILATED label map: {out_path}\n")

print("Label counts (value: pixels, name):")
for v, c in zip(unique_vals, counts):
    nm = "UNASSIGNED(255)" if v == 255 else CLASSES[v]["name"]
    print(f"  {v:3d}: {c:8d}  -> {nm}")

# print("\nPer-class debug masks saved to:", debug_dir)
