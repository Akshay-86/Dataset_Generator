import cv2
import numpy as np
import os
import random

# =========================================================
# ===================== CONFIGURATION =====================
# =========================================================

# ---------- PATHS ----------
FG_PATH = "backend/test/LS20251222120154.png"
BG_PATH = "backend/test/screenshots_20251218_145223_500.00m_1X_p0_v0_a0.png"
OUTPUT_DIR = "assets/testimg"

# ---------- DATASET ----------
DATASET_COUNT = 50

# ---------- SCALE ----------
MIN_SCALE_RATIO = 0.001
START_SCALE_FACTOR = 0.5

# ---------- FILTER LIMITS ----------
FILTER_LIMITS = {
    "brightness": (-100, 100),
    "contrast": (0.5, 2.0),
    "blur": (0, 15)
}

# ---------- GENERATION (GLOBAL IMAGE VARIATION) ----------
GLOBAL_RANDOM = {
    "brightness": (-40, 40),
    "contrast": (0.8, 1.3),
    "blur": (0, 7)
}

INITIAL_OBJECTS = 3

# =========================================================
# ====================== LOAD IMAGES ======================
# =========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

fg_rgba = cv2.imread(FG_PATH, cv2.IMREAD_UNCHANGED)
bg_original = cv2.imread(BG_PATH)

if fg_rgba is None or bg_original is None:
    raise FileNotFoundError("Foreground or Background not found")

bg_h, bg_w = bg_original.shape[:2]

fg_rgb = fg_rgba[:, :, :3]
fg_alpha = fg_rgba[:, :, 3]
orig_h, orig_w = fg_rgb.shape[:2]

FIT_SCALE = min(bg_w / orig_w, bg_h / orig_h)

# =========================================================
# ====================== OBJECT CLASS =====================
# =========================================================

class ObjectInstance:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.scale = FIT_SCALE * START_SCALE_FACTOR
        self.filters = {"brightness": 0, "contrast": 1.0, "blur": 0}
        self.selected = False

# =========================================================
# ======================= STATE ===========================
# =========================================================

objects = []
selected_target = "background"  # or ObjectInstance
dragging = False
offset_x = offset_y = 0

bg_filters = {"brightness": 0, "contrast": 1.0, "blur": 0}

# =========================================================
# ================== OBJECT CREATION ======================
# =========================================================

def create_random_object():
    scale = FIT_SCALE * START_SCALE_FACTOR
    fw = int(orig_w * scale)
    fh = int(orig_h * scale)

    min_x, max_x = fw // 2, bg_w - fw // 2
    min_y, max_y = fh // 2, bg_h - fh // 2

    if min_x >= max_x or min_y >= max_y:
        x, y = bg_w // 2, bg_h // 2
    else:
        x = random.randint(min_x, max_x)
        y = random.randint(min_y, max_y)

    return ObjectInstance(x, y)

for _ in range(INITIAL_OBJECTS):
    objects.append(create_random_object())

# =========================================================
# ===================== FILTER UTILS ======================
# =========================================================

def apply_filters(img, f):
    out = cv2.convertScaleAbs(img, alpha=f["contrast"], beta=f["brightness"])
    if f["blur"] > 1:
        out = cv2.GaussianBlur(out, (f["blur"], f["blur"]), 0)
    return out

# =========================================================
# ===================== CALLBACKS =========================
# =========================================================

def on_brightness(val):
    tgt = bg_filters if selected_target == "background" else selected_target.filters
    tgt["brightness"] = val + FILTER_LIMITS["brightness"][0]

def on_contrast(val):
    tgt = bg_filters if selected_target == "background" else selected_target.filters
    tgt["contrast"] = val / 100.0

def on_blur(val):
    tgt = bg_filters if selected_target == "background" else selected_target.filters
    tgt["blur"] = val if val % 2 == 1 else val + 1

def on_scale(val):
    if isinstance(selected_target, ObjectInstance):
        scale = max(val / 100.0 * FIT_SCALE, MIN_SCALE_RATIO)
        selected_target.scale = scale

# =========================================================
# ===================== MOUSE =============================
# =========================================================

def mouse_cb(event, mx, my, flags, param):
    global selected_target, dragging, offset_x, offset_y

    if event == cv2.EVENT_LBUTTONDOWN:
        selected_target = "background"
        for obj in objects[::-1]:
            fw = int(orig_w * obj.scale)
            fh = int(orig_h * obj.scale)
            if (obj.x - fw // 2 <= mx <= obj.x + fw // 2 and
                obj.y - fh // 2 <= my <= obj.y + fh // 2):
                selected_target = obj
                dragging = True
                offset_x = mx - obj.x
                offset_y = my - obj.y
                break

    elif event == cv2.EVENT_MOUSEMOVE and dragging and isinstance(selected_target, ObjectInstance):
        selected_target.x = mx - offset_x
        selected_target.y = my - offset_y

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

# =========================================================
# ===================== WINDOW ============================
# =========================================================

cv2.namedWindow("Editor", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Editor", mouse_cb)

cv2.createTrackbar("Brightness", "Editor", 100, 200, on_brightness)
cv2.createTrackbar("Contrast", "Editor", 100, 300, on_contrast)
cv2.createTrackbar("Blur", "Editor", 0, FILTER_LIMITS["blur"][1], on_blur)
cv2.createTrackbar("Scale %", "Editor", 50, 100, on_scale)

# =========================================================
# ================= DATASET GENERATION ====================
# =========================================================

def generate_dataset():
    print("🚀 Generating dataset...")
    for i in range(DATASET_COUNT):
        canvas = apply_filters(bg_original, bg_filters)

        for obj in objects:
            fw, fh = int(orig_w * obj.scale), int(orig_h * obj.scale)
            fg_r = cv2.resize(fg_rgb, (fw, fh))
            a_r = cv2.resize(fg_alpha, (fw, fh))

            fg_blended = apply_filters(fg_r, obj.filters)

            x, y = int(obj.x - fw // 2), int(obj.y - fh // 2)
            roi = canvas[y:y+fh, x:x+fw]
            alpha = (a_r / 255.0)[..., None]

            canvas[y:y+fh, x:x+fw] = alpha * fg_blended + (1 - alpha) * roi

        # FINAL GLOBAL VARIATION (KEY IDEA)
        global_f = {
            "brightness": random.randint(*GLOBAL_RANDOM["brightness"]),
            "contrast": random.uniform(*GLOBAL_RANDOM["contrast"]),
            "blur": random.randint(*GLOBAL_RANDOM["blur"])
        }
        global_f["blur"] = global_f["blur"] if global_f["blur"] % 2 else global_f["blur"] + 1

        final_img = apply_filters(canvas, global_f)

        path = os.path.join(OUTPUT_DIR, f"img_{i:04d}.jpg")
        cv2.imwrite(path, final_img)
        print(path)

    print("✅ Dataset generation done")

# =========================================================
# ====================== MAIN LOOP ========================
# =========================================================

while True:
    canvas = apply_filters(bg_original, bg_filters)

    for obj in objects:
        fw, fh = int(orig_w * obj.scale), int(orig_h * obj.scale)
        fg_r = cv2.resize(fg_rgb, (fw, fh))
        a_r = cv2.resize(fg_alpha, (fw, fh))
        fg_adj = apply_filters(fg_r, obj.filters)

        x, y = int(obj.x - fw // 2), int(obj.y - fh // 2)
        roi = canvas[y:y+fh, x:x+fw]
        alpha = (a_r / 255.0)[..., None]

        canvas[y:y+fh, x:x+fw] = alpha * fg_adj + (1 - alpha) * roi

        if selected_target == obj:
            cv2.rectangle(canvas, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

    cv2.imshow("Editor", canvas)

    key = cv2.waitKey(20) & 0xFF
    if key == 27:
        break
    elif key in (ord('n'), ord('N')):
        objects.append(create_random_object())
    elif key in (ord('d'), ord('D')):
        if isinstance(selected_target, ObjectInstance):
            objects.remove(selected_target)
            selected_target = "background"
    elif key in (ord('g'), ord('G')):
        generate_dataset()

cv2.destroyAllWindows()
