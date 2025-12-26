import cv2
import numpy as np
import os
import random

# =========================================================
# ===================== CONFIGURATION =====================
# =========================================================

# ---------- PATHS ----------
FG_PATH = "backend/test/LS20251222120154.png"      # RGBA foreground
BG_PATH = "backend/test/screenshots_20251218_145223_500.00m_1X_p0_v0_a0.png"
OUTPUT_DIR = "backend/test/dataset_output"

# ---------- DATASET ----------
DATASET_COUNT = 50        # number of images to generate

# ---------- FOREGROUND SCALE ----------
MIN_SCALE_RATIO = 0.001   # 0.1%
START_SCALE_FACTOR = 0.5  # half of fit-to-frame size

# ---------- FOREGROUND RANDOMIZATION (GENERATION MODE) ----------
FG_RANDOM = {
    "brightness": (-40, 40),
    "contrast": (0.7, 1.4),
    "blur": (0, 7)   # odd enforced
}

# ---------- BACKGROUND ADJUSTMENT LIMITS (SETUP MODE) ----------
BG_LIMITS = {
    "brightness": (-100, 100),
    "contrast": (0.5, 2.0),
    "blur": (0, 15)
}

# ---------- UI ----------
INITIAL_OBJECTS = 3

# =========================================================
# ====================== LOAD IMAGES ======================
# =========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

fg_rgba = cv2.imread(FG_PATH, cv2.IMREAD_UNCHANGED)
bg_original = cv2.imread(BG_PATH)

if fg_rgba is None or bg_original is None:
    raise FileNotFoundError("Foreground or background not found")

bg_h, bg_w = bg_original.shape[:2]

fg_rgb = fg_rgba[:, :, :3]
fg_alpha = fg_rgba[:, :, 3]
orig_h, orig_w = fg_rgb.shape[:2]

# ---------- FIT SCALE ----------
FIT_SCALE = min(bg_w / orig_w, bg_h / orig_h)

# =========================================================
# ====================== OBJECT CLASS =====================
# =========================================================

class ObjectInstance:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.scale = FIT_SCALE * START_SCALE_FACTOR
        self.selected = False

# =========================================================
# ======================= STATE ===========================
# =========================================================

objects = []
selected_obj = None

dragging = False
offset_x = 0
offset_y = 0

# ---------- BACKGROUND STATE ----------
bg_brightness = 0
bg_contrast = 1.0
bg_blur = 0

# =========================================================
# ================== OBJECT CREATION ======================
# =========================================================

def create_random_object():
    scale = FIT_SCALE * START_SCALE_FACTOR

    fw = int(orig_w * scale)
    fh = int(orig_h * scale)

    # Safe placement bounds
    min_x = fw // 2
    max_x = bg_w - fw // 2
    min_y = fh // 2
    max_y = bg_h - fh // 2

    # Fallback: center if still impossible
    if min_x >= max_x or min_y >= max_y:
        x = bg_w // 2
        y = bg_h // 2
    else:
        x = random.randint(min_x, max_x)
        y = random.randint(min_y, max_y)

    obj = ObjectInstance(x, y)
    obj.scale = scale
    return obj


for _ in range(INITIAL_OBJECTS):
    objects.append(create_random_object())

# =========================================================
# ===================== CALLBACKS =========================
# =========================================================

def on_scale(val):
    if not selected_obj:
        return

    percent = max(val, 1) / 100.0
    scale = percent * FIT_SCALE
    scale = max(scale, MIN_SCALE_RATIO)
    selected_obj.scale = scale

def on_bg_brightness(val):
    global bg_brightness
    bg_brightness = val + BG_LIMITS["brightness"][0]

def on_bg_contrast(val):
    global bg_contrast
    bg_contrast = val / 100.0

def on_bg_blur(val):
    global bg_blur
    bg_blur = val if val % 2 == 1 else val + 1

# =========================================================
# ===================== MOUSE =============================
# =========================================================

def mouse_cb(event, mx, my, flags, param):
    global selected_obj, dragging, offset_x, offset_y

    if event == cv2.EVENT_LBUTTONDOWN:
        selected_obj = None

        for obj in objects[::-1]:
            fw = int(orig_w * obj.scale)
            fh = int(orig_h * obj.scale)

            x0 = obj.x - fw // 2
            y0 = obj.y - fh // 2
            x1 = x0 + fw
            y1 = y0 + fh

            if x0 <= mx <= x1 and y0 <= my <= y1:
                for o in objects:
                    o.selected = False

                obj.selected = True
                selected_obj = obj
                dragging = True
                offset_x = mx - obj.x
                offset_y = my - obj.y

                percent = int((obj.scale / FIT_SCALE) * 100)
                cv2.setTrackbarPos("Scale %", "Editor", max(percent, 1))
                break

    elif event == cv2.EVENT_MOUSEMOVE and dragging and selected_obj:
        selected_obj.x = mx - offset_x
        selected_obj.y = my - offset_y

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

# =========================================================
# ===================== WINDOW ============================
# =========================================================

cv2.namedWindow("Editor", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Editor", mouse_cb)

cv2.createTrackbar("Scale %", "Editor", 50, 100, on_scale)
cv2.createTrackbar("BG Brightness", "Editor", 100, 200, on_bg_brightness)
cv2.createTrackbar("BG Contrast", "Editor", 100, 300, on_bg_contrast)
cv2.createTrackbar("BG Blur", "Editor", 0, BG_LIMITS["blur"][1], on_bg_blur)

# =========================================================
# ================= DATASET GENERATION ====================
# =========================================================

def generate_dataset():
    print("🚀 Generating dataset...")

    for i in range(DATASET_COUNT):
        canvas = cv2.convertScaleAbs(
            bg_original,
            alpha=bg_contrast,
            beta=bg_brightness
        )

        if bg_blur > 1:
            canvas = cv2.GaussianBlur(canvas, (bg_blur, bg_blur), 0)

        for obj in objects:
            fw = int(orig_w * obj.scale)
            fh = int(orig_h * obj.scale)

            fg_r = cv2.resize(fg_rgb, (fw, fh), cv2.INTER_LANCZOS4)
            a_r = cv2.resize(fg_alpha, (fw, fh), cv2.INTER_LINEAR)

            b = random.randint(*FG_RANDOM["brightness"])
            c = random.uniform(*FG_RANDOM["contrast"])
            blur = random.randint(*FG_RANDOM["blur"])
            blur = blur if blur % 2 == 1 else blur + 1

            fg_adj = cv2.convertScaleAbs(fg_r, alpha=c, beta=b)
            if blur > 1:
                fg_adj = cv2.GaussianBlur(fg_adj, (blur, blur), 0)

            x = int(obj.x - fw // 2)
            y = int(obj.y - fh // 2)

            roi = canvas[y:y+fh, x:x+fw]
            alpha = (a_r / 255.0)[..., None]

            canvas[y:y+fh, x:x+fw] = (
                alpha * fg_adj + (1 - alpha) * roi
            ).astype(np.uint8)

        out_path = os.path.join(OUTPUT_DIR, f"img_{i:04d}.jpg")
        cv2.imwrite(out_path, canvas)

    print("✅ Dataset generation completed")

# =========================================================
# ====================== MAIN LOOP ========================
# =========================================================

while True:
    canvas = cv2.convertScaleAbs(
        bg_original,
        alpha=bg_contrast,
        beta=bg_brightness
    )

    if bg_blur > 1:
        canvas = cv2.GaussianBlur(canvas, (bg_blur, bg_blur), 0)

    for obj in objects:
        fw = int(orig_w * obj.scale)
        fh = int(orig_h * obj.scale)

        obj.x = max(fw // 2, min(obj.x, bg_w - fw // 2))
        obj.y = max(fh // 2, min(obj.y, bg_h - fh // 2))

        fg_r = cv2.resize(fg_rgb, (fw, fh), cv2.INTER_LANCZOS4)
        a_r = cv2.resize(fg_alpha, (fw, fh), cv2.INTER_LINEAR)

        x = int(obj.x - fw // 2)
        y = int(obj.y - fh // 2)

        roi = canvas[y:y+fh, x:x+fw]
        alpha = (a_r / 255.0)[..., None]

        canvas[y:y+fh, x:x+fw] = (
            alpha * fg_r + (1 - alpha) * roi
        ).astype(np.uint8)

        if obj.selected:
            cv2.rectangle(
                canvas,
                (x, y),
                (x + fw, y + fh),
                (0, 255, 0),
                2
            )

    cv2.imshow("Editor", canvas)

    key = cv2.waitKey(20) & 0xFF

    if key == 27:
        break
    elif key in (ord('n'), ord('N')):
        objects.append(create_random_object())
    elif key in (ord('d'), ord('D')):
        objects[:] = [o for o in objects if not o.selected]
    elif key in (ord('g'), ord('G')):
        generate_dataset()

cv2.destroyAllWindows()
