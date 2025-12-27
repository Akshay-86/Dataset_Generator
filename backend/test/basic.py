"""
Dataset / Scene Editor & Generator
- Fixed crashes when objects go partially/outside the background by robust clipping.
- Press 'D' to delete the selected instance.
- Supports multiple foreground images (PNG RGBA) from FG_DIR.
- Supports multiple instances of multiple foreground types.
- Click visible (non-transparent) pixels of an instance to select it (clicking transparent area won't select).
- Click empty/background area to select background for adjustments.
- Trackbars adjust selected object (FG) or background (BG) properties.
- Press 'N' to add a new instance (of current foreground type) at random safe position.
- Press 'T' to cycle through foreground types (visible name printed).
- Press 'G' to generate a dataset using FG_RANDOM ranges (uses current background adjustments and instances' positions).
- Press 'S' to save current canvas image (auto-increment filenames).
- ESC to exit.
Paste this file and run in your environment. Adjust CONFIG section paths and values as needed.
"""

import cv2
import numpy as np
import os
import glob
import random
from typing import List

# ---------------------- CONFIG ----------------------
FG_DIR = "assets/foreground"   # directory containing RGBA PNGs (multiple foreground types)
BG_PATH = "assets/background/background.png"
OUTPUT_DIR = "assets/testimg"
SAVE_ONE_PATH = "captured_image.jpg"

DATASET_COUNT = 50

# Foreground initial scale relative to fit-to-frame (0..1)
START_SCALE_FACTOR = 0.5

# Absolute min scale ratio (to background dim)
MIN_SCALE_RATIO = 0.001

# FG randomization ranges used in generation (applied per-instance per-image)
FG_RANDOM = {
    "brightness": (-40, 40),
    "contrast": (0.7, 1.4),
    "blur": (0, 7)  # blur kernel; will be forced to odd
}

# Background adjustment limits (UI trackbars operate in a mapped range)
BG_LIMITS = {
    "brightness": (-100, 100),
    "contrast": (0.5, 2.0),
    "blur": (0, 15)
}

# Initial number of instances spawned at run
INITIAL_OBJECTS = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)
# -----------------------------------------------------

# ---------------------- UTIL -------------------------
def list_pngs(path):
    return sorted(glob.glob(os.path.join(path, "*.png")))

def odd_or_next(v: int) -> int:
    v = int(v)
    if v <= 1:
        return 0
    return v if v % 2 == 1 else v + 1

def safe_write_incremental(output_dir, base="img", ext=".jpg"):
    i = 0
    while True:
        name = f"{base}_{i:04d}{ext}"
        path = os.path.join(output_dir, name)
        if not os.path.exists(path):
            return path
        i += 1

# -----------------------------------------------------

# ---------------------- LOAD -------------------------
# Load background
bg_original = cv2.imread(BG_PATH)
if bg_original is None:
    raise FileNotFoundError(f"Background not found at '{BG_PATH}'")
bg_h, bg_w = bg_original.shape[:2]

# Load all foreground PNGs from FG_DIR (RGBA)
fg_paths = list_pngs(FG_DIR)
if not fg_paths:
    # fallback: try single PNG near the bg folder
    raise FileNotFoundError(f"No PNGs found in FG_DIR='{FG_DIR}'. Put one or more RGBA PNGs there.")

fg_list_rgb: List[np.ndarray] = []
fg_list_alpha: List[np.ndarray] = []
fg_names: List[str] = []
orig_sizes = []

for p in fg_paths:
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("warning: couldn't load", p)
        continue
    if img.shape[2] == 4:
        rgb = img[:, :, :3].copy()
        alpha = img[:, :, 3].copy()
    else:
        # If PNG lacks alpha, create alpha from non-green / non-white (fallback)
        rgb = img[:, :3].copy()
        alpha = 255 * np.ones((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)

    fg_list_rgb.append(rgb)
    fg_list_alpha.append(alpha)
    fg_names.append(os.path.basename(p))
    orig_sizes.append((rgb.shape[1], rgb.shape[0]))  # (w,h)

if not fg_list_rgb:
    raise FileNotFoundError("No valid RGBA foreground images loaded.")

# Compute FIT_SCALE per foreground (fit into background)
FIT_SCALES = []
for (ow, oh) in orig_sizes:
    FIT_SCALES.append(min(bg_w / ow, bg_h / oh))

# current foreground index (for creating new instances)
current_fg_index = 0

# -----------------------------------------------------

# ---------------- OBJECT MODEL ----------------------
class ObjectInstance:
    def __init__(self, fg_index: int, x: int, y: int, scale: float):
        self.fg_index = fg_index  # index into fg_list_*
        self.x = int(x)
        self.y = int(y)
        self.scale = float(scale)
        # per-instance adjustments (these act as "base" adjustments that can be tuned)
        self.brightness = 0      # -100..100 (additive)
        self.contrast = 1.0      # multiplier
        self.blur = 0            # odd kernel
        self.selected = False

    def copy(self):
        o = ObjectInstance(self.fg_index, self.x, self.y, self.scale)
        o.brightness = self.brightness
        o.contrast = self.contrast
        o.blur = self.blur
        o.selected = self.selected
        return o

# Helper to create a random instance for a given fg index
def create_random_instance(fg_index: int) -> ObjectInstance:
    fit = FIT_SCALES[fg_index]
    scale = fit * START_SCALE_FACTOR
    ow, oh = orig_sizes[fg_index]
    fw = max(1, int(ow * scale))
    fh = max(1, int(oh * scale))

    min_x = fw // 2
    max_x = max(min_x, bg_w - fw // 2)
    min_y = fh // 2
    max_y = max(min_y, bg_h - fh // 2)

    if min_x >= max_x or min_y >= max_y:
        x = bg_w // 2
        y = bg_h // 2
    else:
        x = random.randint(min_x, max_x)
        y = random.randint(min_y, max_y)

    return ObjectInstance(fg_index, x, y, scale)

# Start with a few instances (mixed foreground types if multiple)
objects: List[ObjectInstance] = []
for i in range(INITIAL_OBJECTS):
    idx = i % len(fg_list_rgb)
    objects.append(create_random_instance(idx))

selected_obj: ObjectInstance | None = None

# ---------------- BACKGROUND STATE --------------------
bg_brightness = 0
bg_contrast = 1.0
bg_blur = 0

# ---------------- SAVE STATE --------------------------
save_counter = 0

# ---------------- UI / TRACKBARS ---------------------
WINDOW = "Editor"
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

# Trackbar callbacks (they update global state or selected_obj)
def tb_scale_cb(val):
    global selected_obj
    if not selected_obj:
        return
    # val is percent from 1..max_percent
    percent = max(val, 1) / 100.0
    fit = FIT_SCALES[selected_obj.fg_index]
    scale = percent * fit
    selected_obj.scale = max(scale, MIN_SCALE_RATIO)

def tb_fg_brightness_cb(val):
    global selected_obj
    if not selected_obj:
        return
    selected_obj.brightness = val - 100

def tb_fg_contrast_cb(val):
    global selected_obj
    if not selected_obj:
        return
    selected_obj.contrast = val / 100.0

def tb_fg_blur_cb(val):
    global selected_obj
    if not selected_obj:
        return
    selected_obj.blur = odd_or_next(val)

def tb_bg_brightness_cb(val):
    global bg_brightness
    # map 0..200 -> BG_LIMITS["brightness"]
    minv, maxv = BG_LIMITS["brightness"]
    bg_brightness = int(minv + (val / 200.0) * (maxv - minv))

def tb_bg_contrast_cb(val):
    global bg_contrast
    bg_contrast = val / 100.0

def tb_bg_blur_cb(val):
    global bg_blur
    bg_blur = odd_or_next(val)

# Create UI controls
MAX_PERCENT = 100
cv2.createTrackbar("Scale %", WINDOW, int(START_SCALE_FACTOR * 100), MAX_PERCENT, tb_scale_cb)
cv2.createTrackbar("FG Brightness", WINDOW, 100, 200, tb_fg_brightness_cb)
cv2.createTrackbar("FG Contrast", WINDOW, 100, 300, tb_fg_contrast_cb)
cv2.createTrackbar("FG Blur", WINDOW, 0, 25, tb_fg_blur_cb)

cv2.createTrackbar("BG Brightness", WINDOW, 100, 200, tb_bg_brightness_cb)
cv2.createTrackbar("BG Contrast", WINDOW, 100, 300, tb_bg_contrast_cb)
cv2.createTrackbar("BG Blur", WINDOW, 0, BG_LIMITS["blur"][1] if BG_LIMITS["blur"][1] > 0 else 1, tb_bg_blur_cb)

# ---------------- MOUSE CALLBACK ----------------------
dragging = False
drag_offset_x = 0
drag_offset_y = 0

def mouse_cb(event, mx, my, flags, param):
    global selected_obj, dragging, drag_offset_x, drag_offset_y

    if event == cv2.EVENT_LBUTTONDOWN:
        # Check objects in reverse drawing order (topmost first)
        found = False
        for obj in objects[::-1]:
            fg_idx = obj.fg_index
            ow, oh = orig_sizes[fg_idx]
            fw = int(max(1, ow * obj.scale))
            fh = int(max(1, oh * obj.scale))
            x0 = int(obj.x - fw // 2)
            y0 = int(obj.y - fh // 2)
            x1 = x0 + fw
            y1 = y0 + fh
            if x0 <= mx <= x1 and y0 <= my <= y1:
                # compute corresponding alpha pixel in fg (account for clipping)
                fx = mx - x0
                fy = my - y0
                if 0 <= fx < fw and 0 <= fy < fh:
                    # sample alpha by resizing alpha to (fw,fh) then checking pixel
                    alpha = fg_list_alpha[fg_idx]
                    # compute corresponding coords in original alpha
                    orig_w, orig_h = orig_sizes[fg_idx]
                    # map (fx,fy) in resized to original coordinate
                    ox = int((fx / fw) * orig_w)
                    oy = int((fy / fh) * orig_h)
                    ox = min(max(0, ox), orig_w - 1)
                    oy = min(max(0, oy), orig_h - 1)
                    a_val = int(fg_list_alpha[fg_idx][oy, ox])
                    if a_val > 8:
                        # valid click on visible pixel
                        for o in objects:
                            o.selected = False
                        obj.selected = True
                        selected_obj = obj
                        dragging = True
                        drag_offset_x = mx - obj.x
                        drag_offset_y = my - obj.y
                        # Sync trackbars to selected object's values
                        percent = int((obj.scale / FIT_SCALES[obj.fg_index]) * 100)
                        percent = max(1, min(percent, MAX_PERCENT))
                        cv2.setTrackbarPos("Scale %", WINDOW, percent)
                        cv2.setTrackbarPos("FG Brightness", WINDOW, obj.brightness + 100)
                        cv2.setTrackbarPos("FG Contrast", WINDOW, int(obj.contrast * 100))
                        cv2.setTrackbarPos("FG Blur", WINDOW, obj.blur)
                        found = True
                        break
                    # else: clicked transparent region => ignore this object, keep searching
        if not found:
            # Clicked background (no object). Deselect all and set selected_obj=None (so BG trackbars apply)
            for o in objects:
                o.selected = False
            selected_obj = None
            dragging = False
            # sync BG trackbars to current BG state
            # Map bg_brightness from BG_LIMITS to 0..200
            minv, maxv = BG_LIMITS["brightness"]
            if maxv - minv != 0:
                pos = int(((bg_brightness - minv) / (maxv - minv)) * 200)
                pos = max(0, min(200, pos))
                cv2.setTrackbarPos("BG Brightness", WINDOW, pos)
            cv2.setTrackbarPos("BG Contrast", WINDOW, int(bg_contrast * 100))
            cv2.setTrackbarPos("BG Blur", WINDOW, bg_blur)

    elif event == cv2.EVENT_MOUSEMOVE and dragging and selected_obj:
        # Update object's position but do not allow it to become NaN; we allow it to move partially outside,
        # blending will handle clipping safely.
        selected_obj.x = int(mx - drag_offset_x)
        selected_obj.y = int(my - drag_offset_y)

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

cv2.setMouseCallback(WINDOW, mouse_cb)

# ---------------- BLENDING (robust clipping) ----------------
def blend_foreground_onto_canvas(canvas: np.ndarray,
                                 fg_rgb: np.ndarray,
                                 fg_alpha: np.ndarray,
                                 cx: int, cy: int,
                                 scale: float,
                                 fg_adj_brightness: int = 0,
                                 fg_adj_contrast: float = 1.0,
                                 fg_adj_blur: int = 0):
    """
    Blend a RGBA foreground into canvas at center (cx,cy) with given scale.
    Handles partial/out-of-bounds placement safely.
    fg_rgb, fg_alpha are original-sized arrays (orig_h, orig_w).
    Returns: None (canvas modified in place).
    """
    orig_h, orig_w = fg_rgb.shape[:2]
    fw = max(1, int(orig_w * scale))
    fh = max(1, int(orig_h * scale))

    # Resize fg and alpha
    try:
        fg_r = cv2.resize(fg_rgb, (fw, fh), interpolation=cv2.INTER_LANCZOS4)
        a_r = cv2.resize(fg_alpha, (fw, fh), interpolation=cv2.INTER_LINEAR)
    except Exception as e:
        # fallback: skip if resize fails
        return

    # Apply instance adjustments to fg_r
    if fg_adj_contrast != 1.0 or fg_adj_brightness != 0:
        fg_r = cv2.convertScaleAbs(fg_r, alpha=float(fg_adj_contrast), beta=int(fg_adj_brightness))
    if fg_adj_blur and fg_adj_blur > 1:
        fg_r = cv2.GaussianBlur(fg_r, (fg_adj_blur, fg_adj_blur), 0)

    # Compute dst region on canvas
    x0 = int(cx - fw // 2)
    y0 = int(cy - fh // 2)
    x1 = x0 + fw
    y1 = y0 + fh

    # Clip to canvas
    cx0 = max(0, x0)
    cy0 = max(0, y0)
    cx1 = min(canvas.shape[1], x1)
    cy1 = min(canvas.shape[0], y1)

    if cx0 >= cx1 or cy0 >= cy1:
        # fully outside
        return

    # Corresponding region in fg arrays
    fx0 = cx0 - x0
    fy0 = cy0 - y0
    fx1 = fx0 + (cx1 - cx0)
    fy1 = fy0 + (cy1 - cy0)

    fg_part = fg_r[fy0:fy1, fx0:fx1]
    alpha_part = a_r[fy0:fy1, fx0:fx1].astype(np.float32) / 255.0
    alpha_part = alpha_part[..., None]  # shape (h,w,1)

    roi = canvas[cy0:cy1, cx0:cx1].astype(np.float32)

    # Blend
    out = (alpha_part * fg_part.astype(np.float32) + (1.0 - alpha_part) * roi)
    canvas[cy0:cy1, cx0:cx1] = np.clip(out, 0, 255).astype(np.uint8)

# ---------------- GENERATE DATASET --------------------
def generate_dataset():
    print("🚀 Generating dataset...")
    global save_counter
    for i in range(DATASET_COUNT):
        # Apply background adjustments
        canvas = cv2.convertScaleAbs(bg_original, alpha=float(bg_contrast), beta=int(bg_brightness))
        if bg_blur and bg_blur > 1:
            canvas = cv2.GaussianBlur(canvas, (bg_blur, bg_blur), 0)

        # Place each instance with random FG variations
        for obj in objects:
            fg_idx = obj.fg_index
            base_rgb = fg_list_rgb[fg_idx]
            base_alpha = fg_list_alpha[fg_idx]

            # Randomize per-specified FG_RANDOM
            b = random.randint(*FG_RANDOM["brightness"])
            c = random.uniform(*FG_RANDOM["contrast"])
            blur = random.randint(*FG_RANDOM["blur"])
            blur = odd_or_next(blur)

            # combine per-instance base adjustments with random generation offsets
            final_brightness = int(obj.brightness + b)
            final_contrast = float(obj.contrast * c)
            final_blur = obj.blur if obj.blur > 1 else blur

            blend_foreground_onto_canvas(
                canvas,
                base_rgb,
                base_alpha,
                obj.x,
                obj.y,
                obj.scale,
                fg_adj_brightness=final_brightness,
                fg_adj_contrast=final_contrast,
                fg_adj_blur=final_blur
            )

        out_path = os.path.join(OUTPUT_DIR, f"img_{i:04d}.jpg")
        cv2.imwrite(out_path, canvas)
        if i % 10 == 0:
            print(f"  saved {out_path}")
    print("✅ Dataset generation completed. Saved to:", OUTPUT_DIR)

# ---------------- MAIN LOOP ------------------------------
print("Loaded foregrounds:")
for i, name in enumerate(fg_names):
    print(f"  [{i}] {name} (orig {orig_sizes[i][0]}x{orig_sizes[i][1]})")

print("\nControls:")
print("  - Click visible part of an object to select it (click background to select background).")
print("  - Drag selected object to move.")
print("  - Trackbars: when object selected -> FG trackbars control that object; when background selected -> BG trackbars control background.")
print("  - N : add a new instance of current FG type")
print("  - T : cycle current FG type (for new instances)")
print("  - D : delete selected instance")
print("  - G : generate dataset")
print("  - S : save current canvas image snapshot")
print("  - ESC : quit")
print()

cv2.imshow(WINDOW, bg_original)
while True:
    # Prepare canvas with background adjustments
    canvas = cv2.convertScaleAbs(bg_original, alpha=float(bg_contrast), beta=int(bg_brightness))
    if bg_blur and bg_blur > 1:
        canvas = cv2.GaussianBlur(canvas, (bg_blur, bg_blur), 0)

    # Draw objects
    for obj in objects:
        fg_idx = obj.fg_index
        fg_rgb = fg_list_rgb[fg_idx]
        fg_alpha = fg_list_alpha[fg_idx]

        # Clip object's center so it doesn't go completely ridiculous; still allow partial outside
        # (we avoid NaNs but allow user to position outside)
        if not isinstance(obj.x, int):
            obj.x = int(obj.x)
        if not isinstance(obj.y, int):
            obj.y = int(obj.y)

        # Blend with per-instance adjustments (no randomness in preview)
        blend_foreground_onto_canvas(
            canvas, fg_rgb, fg_alpha,
            obj.x, obj.y, obj.scale,
            fg_adj_brightness=obj.brightness,
            fg_adj_contrast=obj.contrast,
            fg_adj_blur=obj.blur
        )

        # Draw selection rect if selected (compute bounds clipped)
        if obj.selected:
            ow, oh = orig_sizes[fg_idx]
            fw = max(1, int(ow * obj.scale))
            fh = max(1, int(oh * obj.scale))
            x0 = int(obj.x - fw // 2)
            y0 = int(obj.y - fh // 2)
            x1 = x0 + fw
            y1 = y0 + fh

            # clip for drawing to canvas coordinates
            cx0 = max(0, x0)
            cy0 = max(0, y0)
            cx1 = min(canvas.shape[1], x1)
            cy1 = min(canvas.shape[0], y1)
            cv2.rectangle(canvas, (cx0, cy0), (cx1, cy1), (0, 255, 0), 2)

    # UI overlay: show current fg type for adding
    label = f"Current FG index: {current_fg_index} / {len(fg_list_rgb)-1}  ({fg_names[current_fg_index]})"
    cv2.putText(canvas, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)

    cv2.imshow(WINDOW, canvas)
    key = cv2.waitKey(20) & 0xFF

    if key == 27:  # ESC
        break
    elif key in (ord('n'), ord('N')):
        # add new instance of current fg type
        objects.append(create_random_instance(current_fg_index))
        print("Added instance of fg index", current_fg_index)
    elif key in (ord('t'), ord('T')):
        current_fg_index = (current_fg_index + 1) % len(fg_list_rgb)
        print("Switched current FG ->", current_fg_index, fg_names[current_fg_index])
    elif key in (ord('d'), ord('D')):
        # delete any selected instances
        before = len(objects)
        objects = [o for o in objects if not o.selected]
        if len(objects) < before:
            print("Deleted selected instance(s)")
        selected_obj = None
    elif key in (ord('s'), ord('S')):
        # save snapshot
        path = safe_write_incremental(OUTPUT_DIR, base="snapshot", ext=".jpg")
        cv2.imwrite(path, canvas)
        print("Saved snapshot:", path)
    elif key in (ord('g'), ord('G')):
        generate_dataset()
    elif key == ord('>') or key == ord('.'):
        # increase BG blur quickly
        bg_blur = odd_or_next(bg_blur + 2)
        cv2.setTrackbarPos("BG Blur", WINDOW, bg_blur)
    elif key == ord('<') or key == ord(','):
        bg_blur = max(0, odd_or_next(bg_blur - 2))
        cv2.setTrackbarPos("BG Blur", WINDOW, bg_blur)

cv2.destroyAllWindows()
