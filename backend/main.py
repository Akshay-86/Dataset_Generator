import cv2
import numpy as np
import os
import random
import json

# =========================================================
# CONFIGURATION
# =========================================================

try:
    with open("backend/data.json","r") as f:
        data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as e:
    print("Failed to load data.json:", e)
    raise SystemExit(1)

# BG_PATH = data["background"]

# =========================================================
# BACKGROUND SWITCH PLUGIN
# =========================================================

BACKGROUND_LIST = [
        "assets/Background/desert/CbivdLKKTLVsjak9RDT9J5-1920-80.jpg",
        "assets/Background/desert/Desert-U.S.webp",
        "assets/Background/desert/libia-sahara-desert-ubari.jpg",
        "assets/Background/forest/india-forest.webp",
        "assets/Background/forest/photo-1542202229-7d93c33f5d07.avif",
        "assets/Background/forest/photo-1441974231531-c6227db76b6e.avif",
        "assets/Background/forest/photo-1508088268825-90a536e9364a.avif",
        "assets/Background/forest/photo0jpg.jpg",
        "assets/Background/forest/Picture-2.jpg",
        "assets/Background/forest/premium_photo-1683444545165-877d0ab2b861.jpeg",
        "assets/Background/forest/shutterstock_601970732.webp",
        "assets/Background/global/background.png",
        "assets/Background/global/clear_road.jpeg",
        "assets/Background/hills/free-photo-of-green-plains-and-hills.jpeg",
        "assets/Background/hills/free-photo-of-landscape-of-hills-and-field.jpeg",
        "assets/Background/plains/gettyimages-1634336092-2048x2048.jpg",
        "assets/Background/plains/istockphoto-905757300-2048x2048.webp",
        "assets/Background/plains/istockphoto-1199401513-612x612.jpg",
        "assets/Background/plains/photo-1618101554052-5b079e68a4bd.jpeg",
        "assets/Background/plains/wide-angle-shot-mountains-trees-foggy-day.jpg"
]

bg_index = 0
is_syncing_ui = False


FOREGROUNDS = {
    name: {
        "path": cfg["path"],    
        "class_id": cfg["class_id"]
    }
    for name, cfg in data["objects"].items()
}

SAVE_DIR = data["save_dir"]["path"]

JSON_DIR = os.path.join(SAVE_DIR, "json")
YOLO_DIR = os.path.join(SAVE_DIR, "yolo_label")

os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(YOLO_DIR, exist_ok=True)

START_SCALE_FACTOR = data["starting_scale"]
MIN_HEIGHT_PX = data["min_height_px"]   # or from data.json


# =========================================================
# LOAD BACKGROUND
# =========================================================


# bg_original = cv2.imread(BACKGROUND_LIST)
# if bg_original is None:
#     raise FileNotFoundError("Background not found")

# bg_h, bg_w = bg_original.shape[:2]

bg_filters = {"brightness": 0, "contrast": 1.0, "blur": 0}


def load_background(path):
    img = cv2.imread(path)
    if img is None:
        print("⚠️ Failed to load background:", path)
        return None
    return img


def cycle_background(step):
    global bg_index, bg_original, bg_h, bg_w

    bg_index = (bg_index + step) % len(BACKGROUND_LIST)
    new_bg = load_background(BACKGROUND_LIST[bg_index])

    if new_bg is None:
        return

    bg_original = new_bg
    bg_h, bg_w = bg_original.shape[:2]

    # 🔧 Re-fit existing objects
    for obj in objects:
        fit = min(bg_w / obj.orig_w, bg_h / obj.orig_h)
        obj.scale = min(obj.scale, fit)

        fw = int(obj.orig_w * obj.scale)
        fh = int(obj.orig_h * obj.scale)

        obj.x = max(fw // 2, min(obj.x, bg_w - fw // 2))
        obj.y = max(fh // 2, min(obj.y, bg_h - fh // 2))

    mark_dirty()  # Background change requires redraw
    sync_trackbars_to_target()
    cv2.setTrackbarMax("Height px", "Editor", bg_h)

    print("🖼️ Background:", BACKGROUND_LIST[bg_index])



# =========================================================
# INITIAL BACKGROUND LOAD (REQUIRED)
# =========================================================

bg_original = load_background(BACKGROUND_LIST[bg_index])
if bg_original is None:
    raise SystemExit("Failed to load initial background")

bg_h, bg_w = bg_original.shape[:2]

MAX_HEIGHT_PX = bg_h

# Initialize cached canvas
cached_canvas = bg_original.copy()
needs_redraw = True

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
selected_objects = set()   # multi-selection set
drag_offsets = {}  # per-object offset during multi-drag

# Performance optimization: cache system
cached_canvas = None
needs_redraw = True
object_render_cache = {}  # Cache for rendered objects


# =========================================================
# HELPERS
# =========================================================

def mark_dirty():
    """Mark that the canvas needs to be redrawn"""
    global needs_redraw, object_render_cache
    needs_redraw = True
    object_render_cache.clear()


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


def get_cached_object_render(obj):
    """Get or create a cached rendered version of an object"""
    # Create a cache key based on object properties
    cache_key = (
        id(obj),
        obj.scale,
        obj.filters["brightness"],
        obj.filters["contrast"],
        obj.filters["blur"]
    )
    
    if cache_key in object_render_cache:
        return object_render_cache[cache_key]
    
    # Render the object
    fw = int(obj.orig_w * obj.scale)
    fh = int(obj.orig_h * obj.scale)
    
    fg = cv2.resize(obj.fg_rgb, (fw, fh))
    a = cv2.resize(obj.fg_alpha, (fw, fh))
    fg = apply_filters(fg, obj.filters)
    
    # Cache the result
    object_render_cache[cache_key] = (fg, a, fw, fh)
    return fg, a, fw, fh


def create_random_object():
    obj_name = current_object_name
    fg = FG_DATA[obj_name]

    fit = min(bg_w / fg["w"], bg_h / fg["h"])
    scale = fit * START_SCALE_FACTOR

    fw = int(fg["w"] * scale)
    fh = int(fg["h"] * scale)

    if fw >= bg_w or fh >= bg_h:
        x, y = bg_w // 2, bg_h // 2
    else:
        x = random.randint(fw // 2, bg_w - fw // 2)
        y = random.randint(fh // 2, bg_h - fh // 2)

    return ObjectInstance(obj_name, x, y)


# =========================================================
# CALLBACKS
# =========================================================

def on_brightness(v):
    mark_dirty()
    if selected_objects:
        for obj in selected_objects:
            obj.filters["brightness"] = v - 100
    else:
        bg_filters["brightness"] = v - 100


def on_contrast(v):
    mark_dirty()
    if selected_objects:
        for obj in selected_objects:
            obj.filters["contrast"] = v / 100.0
    else:
        bg_filters["contrast"] = v / 100.0

def on_blur(v):
    mark_dirty()
    v = v if v % 2 else v + 1
    if selected_objects:
        for obj in selected_objects:
            obj.filters["blur"] = v
    else:
        bg_filters["blur"] = v


def on_height_change(val):
    global is_syncing_ui

    if is_syncing_ui:
        return

    if not selected_objects:
        return

    mark_dirty()
    for obj in selected_objects:
        height_px = max(val, MIN_HEIGHT_PX)
        scale = height_px / obj.orig_h

        max_x = (2 * min(obj.x, bg_w - obj.x)) / obj.orig_w
        max_y = (2 * min(obj.y, bg_h - obj.y)) / obj.orig_h

        obj.scale = min(scale, max_x, max_y)





# =========================================================
# MOUSE
# =========================================================

def mouse_cb(event, mx, my, flags, param):
    global selected_target, dragging, drag_offsets, selected_objects

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = False
        drag_offsets.clear()

        clicked_obj = None
        for obj in objects[::-1]:
            fw = int(obj.orig_w * obj.scale)
            fh = int(obj.orig_h * obj.scale)
            if abs(mx - obj.x) <= fw // 2 and abs(my - obj.y) <= fh // 2:
                clicked_obj = obj
                break

        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            if clicked_obj:
                if clicked_obj in selected_objects:
                    selected_objects.remove(clicked_obj)
                else:
                    selected_objects.add(clicked_obj)
                selected_target = clicked_obj
        else:
            selected_objects.clear()
            if clicked_obj:
                selected_objects.add(clicked_obj)
                selected_target = clicked_obj
            else:
                selected_target = "background"

        if clicked_obj and clicked_obj in selected_objects:
            dragging = True
            base_dx = mx - clicked_obj.x
            base_dy = my - clicked_obj.y

            for obj in selected_objects:
                drag_offsets[obj] = (
                    base_dx + (clicked_obj.x - obj.x),
                    base_dy + (clicked_obj.y - obj.y)
                )

        mark_dirty()  # Selection change requires redraw
        sync_trackbars_to_target()

    elif event == cv2.EVENT_MOUSEMOVE and dragging and selected_objects:
        mark_dirty()  # Movement requires redraw
        for obj in selected_objects:
            dx, dy = drag_offsets[obj]
            fw = int(obj.orig_w * obj.scale)
            fh = int(obj.orig_h * obj.scale)

            obj.x = max(fw // 2, min(mx - dx, bg_w - fw // 2))
            obj.y = max(fh // 2, min(my - dy, bg_h - fh // 2))

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        drag_offsets.clear()




# =========================================================
# SYNC TRACKBARS
# =========================================================

def sync_trackbars_to_target():
    global is_syncing_ui
    is_syncing_ui = True

    tgt = bg_filters if selected_target == "background" else selected_target.filters

    cv2.setTrackbarPos("Brightness", "Editor", tgt["brightness"] + 100)
    cv2.setTrackbarPos("Contrast", "Editor", int(tgt["contrast"] * 100))
    cv2.setTrackbarPos("Blur", "Editor", tgt["blur"])

    if isinstance(selected_target, ObjectInstance):
        height_px = int(selected_target.orig_h * selected_target.scale)
        cv2.setTrackbarPos("Height px", "Editor", height_px)

    is_syncing_ui = False



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


#----------------------------------------------------------
#multilinewrite progran
def putText_multiline(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, color=(255, 255, 255), thickness=1, line_type=cv2.LINE_AA, line_spacing=1.4):

    x, y = org
    lines = text.split('\n')

    # Get height of one line
    (w, h), baseline = cv2.getTextSize(
        "A", font, font_scale, thickness
    )
    line_height = int(h * line_spacing)

    for i, line in enumerate(lines):
        y_i = y + i * line_height
        cv2.putText(img, line, (x, y_i), font, font_scale, color, thickness, line_type)

#--------------------------
def get_next_filename(base="scene"):
    i = 1
    while True:
        name = f"{base}_{i:03d}"
        if not os.path.exists(os.path.join(SAVE_DIR, name + ".png")):
            return name
        i += 1


# =========================================================
# WINDOW
# =========================================================

cv2.namedWindow("Editor", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Editor", mouse_cb)

cv2.createTrackbar("Brightness", "Editor", 100, 200, on_brightness)
cv2.createTrackbar("Contrast", "Editor", 100, 300, on_contrast)
cv2.createTrackbar("Blur", "Editor", 0, 15, on_blur)
cv2.createTrackbar("Height px", "Editor", 100, 2000, on_height_change)

# =========================================================
# MAIN LOOP
# =========================================================

while True:
    # Only redraw if something changed
    if needs_redraw:
        canvas = apply_filters(bg_original.copy(), bg_filters)

        for obj in objects:
            fg, a, fw, fh = get_cached_object_render(obj)
            canvas = safe_blend(canvas, fg, a, obj.x, obj.y)

            if obj in selected_objects:
                color = (0, 255, 0) if obj == selected_target else (255, 0, 0)
                cv2.rectangle(
                    canvas,
                    (obj.x - fw // 2, obj.y - fh // 2),
                    (obj.x + fw // 2, obj.y + fh // 2),
                    color, 2
                )

        if isinstance(selected_target, ObjectInstance):
            fh = int(selected_target.orig_h * selected_target.scale)
            cv2.putText(canvas, f"Height: {fh}px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(canvas, f"Current object: {current_object_name}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        txt="Depth  Height(px)\n 10      177\n 15      118\n 20      89\n 25      75\n 30      66\n 35      56\n 40      47\n 45      44\n 50      42\n 55      36\n 60      35\n 65      33\n 70      29"
        putText_multiline(canvas,txt,(bg_w - 250,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cached_canvas = canvas
        needs_redraw = False

    cv2.imshow("Editor", cached_canvas)

    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('['):
        cycle_object(-1)
    elif k == ord(']'):
        cycle_object(1)
    elif k in (ord('n'), ord('N')):
        objects.append(create_random_object())
        mark_dirty()
    elif k in (ord('d'), ord('D')):
        if isinstance(selected_target, ObjectInstance):
            objects.remove(selected_target)
            selected_target = "background"
            mark_dirty()
    elif k in (ord('s'), ord('S')):
        name = get_next_filename()
        save_combined(name)
    elif k in (ord('c'), ord('C')):
        objects.clear()
        selected_objects.clear()
        selected_target = "background"
        mark_dirty()
        sync_trackbars_to_target()

    elif k == ord(','):   # previous background
        cycle_background(-1)

    elif k == ord('.'):   # next background
        cycle_background(1)



cv2.destroyAllWindows()
