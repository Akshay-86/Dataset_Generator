import cv2
import numpy as np
import os
import random
import json

# =========================================================
# CONFIGURATION
# =========================================================

BG_PATH = "assets/Background/wide-angle-shot-mountains-trees-foggy-day.jpg"

FOREGROUNDS = {
    "object1": {
        "path": "assets/Foreground/fig_11 (1).png",
        "class_id": 81
    },
    "object2": {
        "path": "assets/Foreground/soldier.png",
        "class_id": 82
    }
}

SAVE_DIR = "assets/Combined"
JSON_DIR = os.path.join(SAVE_DIR, "json")
YOLO_DIR = os.path.join(SAVE_DIR, "yolo_label")

os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(YOLO_DIR, exist_ok=True)

MIN_SCALE_RATIO = 0.1
START_SCALE_FACTOR = 0.3

# =========================================================
# LOAD BACKGROUND
# =========================================================

bg_original = cv2.imread(BG_PATH)
if bg_original is None:
    raise FileNotFoundError("Background not found")

bg_h, bg_w = bg_original.shape[:2]
bg_filters = {"brightness": 0, "contrast": 1.0, "blur": 0}

# =========================================================
# LOAD FOREGROUNDS
# =========================================================

FG_DATA = {}
for name, cfg in FOREGROUNDS.items():
    rgba = cv2.imread(cfg["path"], cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.shape[2] != 4:
        raise ValueError(f"{name} must be RGBA")

    FG_DATA[name] = {
        "rgb": rgba[:, :, :3],
        "alpha": rgba[:, :, 3],
        "h": rgba.shape[0],
        "w": rgba.shape[1],
        "class_id": cfg["class_id"]
    }

# =========================================================
# OBJECT INSTANCE
# =========================================================

class ObjectInstance:
    def __init__(self, obj_name, x, y):
        fg = FG_DATA[obj_name]

        self.object_name = obj_name
        self.class_id = fg["class_id"]

        self.fg_rgb = fg["rgb"]
        self.fg_alpha = fg["alpha"]
        self.orig_h = fg["h"]
        self.orig_w = fg["w"]

        fit = min(bg_w / self.orig_w, bg_h / self.orig_h)
        self.scale = fit * START_SCALE_FACTOR

        self.x = x
        self.y = y
        self.filters = {"brightness": 0, "contrast": 1.0, "blur": 0}

# =========================================================
# STATE
# =========================================================

objects = []
selected_target = "background"
dragging = False
offset_x = offset_y = 0

object_names = list(FOREGROUNDS.keys())
current_object_index = 0
current_object_name = object_names[current_object_index]

# =========================================================
# HELPERS
# =========================================================

def cycle_object(step):
    global current_object_index, current_object_name
    current_object_index = (current_object_index + step) % len(object_names)
    current_object_name = object_names[current_object_index]
    print("🟢 Current object:", current_object_name)


def apply_filters(img, f):
    out = cv2.convertScaleAbs(img, alpha=f["contrast"], beta=f["brightness"])
    if f["blur"] > 1:
        k = f["blur"] if f["blur"] % 2 else f["blur"] + 1
        out = cv2.GaussianBlur(out, (k, k), 0)
    return out


def safe_blend(canvas, fg, alpha, cx, cy):
    fh, fw = fg.shape[:2]
    x0, y0 = int(cx - fw // 2), int(cy - fh // 2)
    x1, y1 = x0 + fw, y0 + fh

    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(bg_w, x1), min(bg_h, y1)

    if cx0 >= cx1 or cy0 >= cy1:
        return canvas

    fx0, fy0 = cx0 - x0, cy0 - y0

    roi = canvas[cy0:cy1, cx0:cx1]
    fg_part = fg[fy0:fy0 + (cy1 - cy0), fx0:fx0 + (cx1 - cx0)]
    a_part = (alpha[fy0:fy0 + (cy1 - cy0), fx0:fx0 + (cx1 - cx0)] / 255.0)[..., None]

    canvas[cy0:cy1, cx0:cx1] = (a_part * fg_part + (1 - a_part) * roi).astype(np.uint8)
    return canvas


def create_random_object():
    obj_name = current_object_name
    fg = FG_DATA[obj_name]

    fit = min(bg_w / fg["w"], bg_h / fg["h"])
    scale = fit * START_SCALE_FACTOR

    fw = int(fg["w"] * scale)
    fh = int(fg["h"] * scale)

    x = random.randint(fw // 2, bg_w - fw // 2)
    y = random.randint(fh // 2, bg_h - fh // 2)

    return ObjectInstance(obj_name, x, y)

# =========================================================
# CALLBACKS
# =========================================================

def on_brightness(v):
    tgt = bg_filters if selected_target == "background" else selected_target.filters
    tgt["brightness"] = v - 100

def on_contrast(v):
    tgt = bg_filters if selected_target == "background" else selected_target.filters
    tgt["contrast"] = v / 100.0

def on_blur(v):
    tgt = bg_filters if selected_target == "background" else selected_target.filters
    tgt["blur"] = v if v % 2 else v + 1

def on_scale(v):
    if not isinstance(selected_target, ObjectInstance):
        return

    req = max(v / 100.0, MIN_SCALE_RATIO)

    max_x = (2 * min(selected_target.x, bg_w - selected_target.x)) / selected_target.orig_w
    max_y = (2 * min(selected_target.y, bg_h - selected_target.y)) / selected_target.orig_h

    selected_target.scale = max(
        MIN_SCALE_RATIO,
        min(req, max_x, max_y)
    )

# =========================================================
# MOUSE
# =========================================================

def mouse_cb(event, mx, my, flags, param):
    global selected_target, dragging, offset_x, offset_y

    if event == cv2.EVENT_LBUTTONDOWN:
        selected_target = "background"
        for obj in objects[::-1]:
            fw = int(obj.orig_w * obj.scale)
            fh = int(obj.orig_h * obj.scale)
            if abs(mx - obj.x) <= fw // 2 and abs(my - obj.y) <= fh // 2:
                selected_target = obj
                dragging = True
                offset_x = mx - obj.x
                offset_y = my - obj.y
                break
        sync_trackbars_to_target()

    elif event == cv2.EVENT_MOUSEMOVE and dragging and isinstance(selected_target, ObjectInstance):
        fw = int(selected_target.orig_w * selected_target.scale)
        fh = int(selected_target.orig_h * selected_target.scale)

        selected_target.x = max(fw // 2, min(mx - offset_x, bg_w - fw // 2))
        selected_target.y = max(fh // 2, min(my - offset_y, bg_h - fh // 2))

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

# =========================================================
# SYNC TRACKBARS
# =========================================================

def sync_trackbars_to_target():
    tgt = bg_filters if selected_target == "background" else selected_target.filters

    cv2.setTrackbarPos("Brightness", "Editor", tgt["brightness"] + 100)
    cv2.setTrackbarPos("Contrast", "Editor", int(tgt["contrast"] * 100))
    cv2.setTrackbarPos("Blur", "Editor", tgt["blur"])

    if isinstance(selected_target, ObjectInstance):
        cv2.setTrackbarPos("Scale %", "Editor", int(selected_target.scale * 100))

# =========================================================
# SAVE COMBINED
# =========================================================

def save_combined(name):
    canvas = apply_filters(bg_original.copy(), bg_filters)
    yolo = []
    data = {"background": {"width": bg_w, "height": bg_h}, "foreground": {}}

    for obj in objects:
        data["foreground"].setdefault(obj.object_name, {
            "class_id": obj.class_id,
            "instances": []
        })

        fw = int(obj.orig_w * obj.scale)
        fh = int(obj.orig_h * obj.scale)

        fg = cv2.resize(obj.fg_rgb, (fw, fh))
        a = cv2.resize(obj.fg_alpha, (fw, fh))
        fg = apply_filters(fg, obj.filters)

        canvas = safe_blend(canvas, fg, a, obj.x, obj.y)

        yolo.append(
            f"{obj.class_id} "
            f"{obj.x / bg_w:.6f} "
            f"{obj.y / bg_h:.6f} "
            f"{fw / bg_w:.6f} "
            f"{fh / bg_h:.6f}"
        )

        data["foreground"][obj.object_name]["instances"].append({
            "center": [obj.x, obj.y],
            "scale": obj.scale,
            "filters": obj.filters
        })

    cv2.imwrite(os.path.join(SAVE_DIR, f"{name}.png"), canvas)

    with open(os.path.join(JSON_DIR, f"{name}.json"), "w") as f:
        json.dump(data, f, indent=4)

    with open(os.path.join(YOLO_DIR, f"{name}.txt"), "w") as f:
        f.write("\n".join(yolo))

    print("✅ saved combined:", name)

# =========================================================
# WINDOW
# =========================================================

cv2.namedWindow("Editor", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Editor", mouse_cb)

cv2.createTrackbar("Brightness", "Editor", 100, 200, on_brightness)
cv2.createTrackbar("Contrast", "Editor", 100, 300, on_contrast)
cv2.createTrackbar("Blur", "Editor", 0, 15, on_blur)
cv2.createTrackbar("Scale %", "Editor", int(START_SCALE_FACTOR * 100), 200, on_scale)

# =========================================================
# MAIN LOOP
# =========================================================

while True:
    canvas = apply_filters(bg_original.copy(), bg_filters)

    for obj in objects:
        fw = int(obj.orig_w * obj.scale)
        fh = int(obj.orig_h * obj.scale)

        fg = cv2.resize(obj.fg_rgb, (fw, fh))
        a = cv2.resize(obj.fg_alpha, (fw, fh))
        fg = apply_filters(fg, obj.filters)

        canvas = safe_blend(canvas, fg, a, obj.x, obj.y)

        if selected_target == obj:
            cv2.rectangle(
                canvas,
                (obj.x - fw // 2, obj.y - fh // 2),
                (obj.x + fw // 2, obj.y + fh // 2),
                (0, 255, 0), 2
            )

    cv2.putText(canvas, f"Current object: {current_object_name}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow("Editor", canvas)

    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('['):
        cycle_object(-1)
    elif k == ord(']'):
        cycle_object(1)
    elif k in (ord('n'), ord('N')):
        objects.append(create_random_object())
    elif k in (ord('d'), ord('D')):
        if isinstance(selected_target, ObjectInstance):
            objects.remove(selected_target)
            selected_target = "background"
    elif k in (ord('s'), ord('S')):
        save_combined("forest_1")
    elif k in (ord('c'), ord('C')):
        objects.clear()
        selected_target = "background"


cv2.destroyAllWindows()
