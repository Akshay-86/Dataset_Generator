import cv2
import numpy as np
import os
import random

# =========================================================
# CONFIGURATION
# =========================================================

FG_PATH = "assets/foreground/fig_11.png"
BG_PATH = "assets/background/photo0jpg.jpg"
OUTPUT_DIR = "assets/testimg"

DATASET_COUNT = 20
# ---------- YOLO ----------
YOLO_CLASS_ID = 0          # class index (0 = person, 1 = car, etc.)
YOLO_LABEL_DIR = "assets/testimg/labels"
os.makedirs(YOLO_LABEL_DIR, exist_ok=True)


MIN_SCALE_RATIO = 0.01
START_SCALE_FACTOR = 0.5

FILTER_LIMITS = {
    "brightness": (-100, 100),
    "contrast": (0.5, 2.0),
    "blur": (0, 15)
}

GLOBAL_RANDOM = {
    "brightness": (-40, 40),
    "contrast": (0.8, 1.3),
    "blur": (0, 7)
}

INITIAL_OBJECTS = 3

# =========================================================
# LOAD
# =========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

fg_rgba = cv2.imread(FG_PATH, cv2.IMREAD_UNCHANGED)
bg_original = cv2.imread(BG_PATH)

if fg_rgba is None or bg_original is None:
    raise FileNotFoundError("Image not found")

bg_h, bg_w = bg_original.shape[:2]

fg_rgb = fg_rgba[:, :, :3]
fg_alpha = fg_rgba[:, :, 3]
orig_h, orig_w = fg_rgb.shape[:2]

FIT_SCALE = min(bg_w / orig_w, bg_h / orig_h)

# =========================================================
# OBJECT
# =========================================================

class ObjectInstance:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.scale = FIT_SCALE * START_SCALE_FACTOR
        self.filters = {"brightness": 0, "contrast": 1.0, "blur": 0}
        self.selected = False

# =========================================================
# STATE
# =========================================================

objects = []
selected_target = "background"
dragging = False
offset_x = offset_y = 0

bg_filters = {"brightness": 0, "contrast": 1.0, "blur": 0}

# =========================================================
# HELPERS
# =========================================================

def apply_filters(img, f):
    out = cv2.convertScaleAbs(img, alpha=f["contrast"], beta=f["brightness"])
    if f["blur"] > 1:
        out = cv2.GaussianBlur(out, (f["blur"], f["blur"]), 0)
    return out


def safe_blend(canvas, fg, alpha, cx, cy):
    fh, fw = fg.shape[:2]

    x0 = int(cx - fw // 2)
    y0 = int(cy - fh // 2)
    x1 = x0 + fw
    y1 = y0 + fh

    cx0 = max(0, x0)
    cy0 = max(0, y0)
    cx1 = min(bg_w, x1)
    cy1 = min(bg_h, y1)

    if cx0 >= cx1 or cy0 >= cy1:
        return canvas

    fx0 = cx0 - x0
    fy0 = cy0 - y0
    fx1 = fx0 + (cx1 - cx0)
    fy1 = fy0 + (cy1 - cy0)

    roi = canvas[cy0:cy1, cx0:cx1]
    fg_part = fg[fy0:fy1, fx0:fx1]
    a_part = (alpha[fy0:fy1, fx0:fx1] / 255.0)[..., None]

    canvas[cy0:cy1, cx0:cx1] = (
        a_part * fg_part + (1 - a_part) * roi
    ).astype(np.uint8)

    return canvas


def create_random_object():
    scale = FIT_SCALE * START_SCALE_FACTOR
    fw = int(orig_w * scale)
    fh = int(orig_h * scale)

    x = random.randint(0, bg_w)
    y = random.randint(0, bg_h)

    obj = ObjectInstance(x, y)
    obj.scale = scale
    return obj


for _ in range(INITIAL_OBJECTS):
    objects.append(create_random_object())

# =========================================================
# CALLBACKS
# =========================================================

def on_brightness(val):
    tgt = bg_filters if selected_target == "background" else selected_target.filters
    tgt["brightness"] = val - 100

def on_contrast(val):
    tgt = bg_filters if selected_target == "background" else selected_target.filters
    tgt["contrast"] = val / 100.0

def on_blur(val):
    tgt = bg_filters if selected_target == "background" else selected_target.filters
    tgt["blur"] = val if val % 2 == 1 else val + 1

def on_scale(val):
    if not isinstance(selected_target, ObjectInstance):
        return

    requested_scale = max(val / 100.0 * FIT_SCALE, MIN_SCALE_RATIO)

    # Max scale allowed so object fits frame at current position
    max_scale_x = (2 * min(selected_target.x, bg_w - selected_target.x)) / orig_w
    max_scale_y = (2 * min(selected_target.y, bg_h - selected_target.y)) / orig_h
    max_allowed_scale = min(max_scale_x, max_scale_y)

    selected_target.scale = min(requested_scale, max_allowed_scale)


# =========================================================
# MOUSE
# =========================================================

def mouse_cb(event, mx, my, flags, param):
    global selected_target, dragging, offset_x, offset_y

    if event == cv2.EVENT_LBUTTONDOWN:
        selected_target = "background"
        for obj in objects[::-1]:
            fw = int(orig_w * obj.scale)
            fh = int(orig_h * obj.scale)
            if abs(mx - obj.x) <= fw // 2 and abs(my - obj.y) <= fh // 2:
                selected_target = obj
                dragging = True
                offset_x = mx - obj.x
                offset_y = my - obj.y
                break
        sync_trackbars_to_target()

    elif event == cv2.EVENT_MOUSEMOVE and dragging and isinstance(selected_target, ObjectInstance):
        fw = int(orig_w * selected_target.scale)
        fh = int(orig_h * selected_target.scale)

        new_x = mx - offset_x
        new_y = my - offset_y

        # Clamp so object never leaves frame
        selected_target.x = max(fw // 2, min(new_x, bg_w - fw // 2))
        selected_target.y = max(fh // 2, min(new_y, bg_h - fh // 2))


    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False


def sync_trackbars_to_target():
    if selected_target == "background":
        cv2.setTrackbarPos(
            "Brightness", "Editor",
            bg_filters["brightness"] - FILTER_LIMITS["brightness"][0]
        )
        cv2.setTrackbarPos(
            "Contrast", "Editor",
            int(bg_filters["contrast"] * 100)
        )
        cv2.setTrackbarPos(
            "Blur", "Editor",
            bg_filters["blur"]
        )
    else:
        cv2.setTrackbarPos(
            "Brightness", "Editor",
            selected_target.filters["brightness"] - FILTER_LIMITS["brightness"][0]
        )
        cv2.setTrackbarPos(
            "Contrast", "Editor",
            int(selected_target.filters["contrast"] * 100)
        )
        cv2.setTrackbarPos(
            "Blur", "Editor",
            selected_target.filters["blur"]
        )
        cv2.setTrackbarPos(
            "Scale %", "Editor",
            int((selected_target.scale / FIT_SCALE) * 100)
        )


# =========================================================
# WINDOW
# =========================================================

cv2.namedWindow("Editor", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Editor", mouse_cb)

cv2.createTrackbar("Brightness", "Editor", 100, 200, on_brightness)
cv2.createTrackbar("Contrast", "Editor", 100, 300, on_contrast)
cv2.createTrackbar("Blur", "Editor", 0, FILTER_LIMITS["blur"][1], on_blur)
cv2.createTrackbar("Scale %", "Editor", 50, 100, on_scale)

# =========================================================
# GENERATION
# =========================================================

def generate_dataset():
    for i in range(DATASET_COUNT):
        # ---------- APPLY BACKGROUND FILTERS ----------
        canvas = apply_filters(bg_original, bg_filters)

        yolo_lines = []

        # ---------- DRAW ALL OBJECTS ----------
        for obj in objects:
            fw = int(orig_w * obj.scale)
            fh = int(orig_h * obj.scale)

            fg_r = cv2.resize(fg_rgb, (fw, fh), cv2.INTER_LANCZOS4)
            a_r = cv2.resize(fg_alpha, (fw, fh), cv2.INTER_LINEAR)

            fg_adj = apply_filters(fg_r, obj.filters)
            canvas = safe_blend(canvas, fg_adj, a_r, obj.x, obj.y)

            # ---------- YOLO BBOX ----------
            x_center = obj.x / bg_w
            y_center = obj.y / bg_h
            w_norm = fw / bg_w
            h_norm = fh / bg_h

            yolo_lines.append(
                f"{YOLO_CLASS_ID} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{w_norm:.6f} "
                f"{h_norm:.6f}"
            )

        # ---------- GLOBAL FILTER RANDOMIZATION ----------
        global_f = {
            "brightness": random.randint(*GLOBAL_RANDOM["brightness"]),
            "contrast": random.uniform(*GLOBAL_RANDOM["contrast"]),
            "blur": random.randint(*GLOBAL_RANDOM["blur"])
        }

        if global_f["blur"] > 0 and global_f["blur"] % 2 == 0:
            global_f["blur"] += 1

        final_img = apply_filters(canvas, global_f)

        # ---------- FILENAME ----------
        b = global_f["brightness"]
        c = int(global_f["contrast"] * 100)
        bl = global_f["blur"]

        img_name = (
            f"brightness{b:+04d}_"
            f"contrast{c:03d}_"
            f"blur{bl:02d}_"
            f"img{i:04d}"
        )

        img_path = os.path.join(OUTPUT_DIR, img_name + ".jpg")
        txt_path = os.path.join(YOLO_LABEL_DIR, img_name + ".txt")

        cv2.imwrite(img_path, final_img)

        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_lines))

        print(img_name)

    print("✅ Dataset + YOLO labels generated")




# =========================================================
# MAIN LOOP
# =========================================================

while True:
    canvas = apply_filters(bg_original, bg_filters)

    for obj in objects:
        fw = int(orig_w * obj.scale)
        fh = int(orig_h * obj.scale)

        obj.x = max(fw // 2, min(obj.x, bg_w - fw // 2))
        obj.y = max(fh // 2, min(obj.y, bg_h - fh // 2))


        fg_r = cv2.resize(fg_rgb, (fw, fh))
        a_r = cv2.resize(fg_alpha, (fw, fh))
        fg_adj = apply_filters(fg_r, obj.filters)

        canvas = safe_blend(canvas, fg_adj, a_r, obj.x, obj.y)

        if selected_target == obj:
            cv2.rectangle(
                canvas,
                (int(obj.x - fw // 2), int(obj.y - fh // 2)),
                (int(obj.x + fw // 2), int(obj.y + fh // 2)),
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
        if isinstance(selected_target, ObjectInstance):
            objects.remove(selected_target)
            selected_target = "background"
    elif key in (ord('g'), ord('G')):
        generate_dataset()

cv2.destroyAllWindows()
