import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Parameters
map_image_path = "blured_map.jpg"   
original_image_path = "original.jpg"
scale = 0.55
depth_range = 40
shore_color_threshold = 20  # color distance threshold for shore masking

# Load images
map_img = cv2.imread(map_image_path)
original_img = cv2.imread(original_image_path)

if map_img is None or original_img is None:
    print("Failed to load one or both images.")
    exit()

# Resize for user input interface
resized_original = cv2.resize(original_img, (0, 0), fx=scale, fy=scale)
resized_map = cv2.resize(map_img, (0, 0), fx=scale, fy=scale)
h, w = resized_original.shape[:2]

click_data = []

# Mouse callback
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_color = resized_map[y, x]
        print(f"Clicked color (BGR): {ref_color}")
        try:
            depth = float(input("Enter depth at this point (cm): "))
            click_data.append((ref_color.astype(np.float32), x, y, depth))
        except:
            print("Invalid input")

cv2.namedWindow("Click on original (ESC to finish)")
cv2.setMouseCallback("Click on original (ESC to finish)", on_mouse)

# Input loop
while True:
    temp = resized_original.copy()
    for _, x, y, _ in click_data:
        cv2.circle(temp, (x, y), 5, (0, 0, 255), -1)
    cv2.imshow("Click on original (ESC to finish)", temp)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()

if len(click_data) < 1:
    print("No input points.")
    exit()

# Identify shore colors from input where depth == 0
shore_colors = [color for color, _, _, depth in click_data if depth == 0]
shore_mask = np.full((h, w), False, dtype=bool)

for shore_color in shore_colors:
    diff = np.linalg.norm(resized_map.astype(np.float32) - shore_color, axis=2)
    shore_mask |= diff < shore_color_threshold

# Compute weighted depth map
diff_stack = []
weight_stack = []

for color, x, y, base_depth in click_data:
    diff = np.linalg.norm(resized_map.astype(np.float32) - color, axis=2)
    norm_diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    depth_map = base_depth - norm_diff * depth_range  # preserve base depth exactly
    weight = 1.0 / (norm_diff + 0.05)
    diff_stack.append(depth_map * weight)
    weight_stack.append(weight)

depth_sum = np.sum(diff_stack, axis=0)
weight_sum = np.sum(weight_stack, axis=0)
final_depth = depth_sum / (weight_sum + 1e-8)
final_depth = np.maximum(final_depth, 0)

# Resize to original image resolution
full_h, full_w = original_img.shape[:2]
depth_resized = cv2.resize(final_depth, (full_w, full_h), interpolation=cv2.INTER_CUBIC)
shore_mask_full = cv2.resize(shore_mask.astype(np.uint8), (full_w, full_h), interpolation=cv2.INTER_NEAREST).astype(bool)

# Mask black + shore areas for contour exclusion
black_mask = np.any(original_img != [255, 255, 255], axis=2)  # after replacing black with white
combined_mask = np.logical_and(black_mask, ~shore_mask_full)
masked_depth = np.where(combined_mask, depth_resized, np.nan)

# Draw contour on original image
plt.figure(figsize=(12, 10))
plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
cs = plt.contour(masked_depth, levels=np.arange(np.nanmin(masked_depth), np.nanmax(masked_depth), 20), cmap=cm.cool)
plt.clabel(cs, inline=True, fontsize=8)
plt.title("Depth Contour on Original Image")
plt.axis('off')
plt.tight_layout()
plt.savefig("contour_on_original.png", dpi=300)
plt.show()
