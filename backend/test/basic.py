import cv2
import numpy as np
import os

# ----------------------------
# CONFIG
# ----------------------------
FG_PATH = "backend/test/LS20251222120154.png"      # RGBA PNG
BG_PATH = "backend/test/screenshots_20251218_145223_500.00m_1X_p0_v0_a0.png"
OUTPUT_PATH = "captured_image.jpg"
MIN_SCALE_RATIO = 0.050   # 0.1%

# ----------------------------
# LOAD IMAGES
# ----------------------------
fg_rgba = cv2.imread(FG_PATH, cv2.IMREAD_UNCHANGED)
bg = cv2.imread(BG_PATH)

if fg_rgba is None or bg is None:
    raise FileNotFoundError("Image not found")

bg_h, bg_w = bg.shape[:2]

fg_rgb = fg_rgba[:, :, :3]
fg_alpha = fg_rgba[:, :, 3]
orig_h, orig_w = fg_rgb.shape[:2]

# ----------------------------
# INITIAL FIT SCALE
# ----------------------------
fit_scale = min(bg_w / orig_w, bg_h / orig_h)
fit_scale = min(fit_scale, 1.0)  # never upscale at start


# ----------------------------
# OBJECT CLASS
# ----------------------------
class ObjectInstance:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.scale = fit_scale
        self.brightness = 0
        self.contrast = 1.0
        self.blur = 0
        self.selected = False

# ----------------------------
# STATE
# ----------------------------
objects = [ObjectInstance(bg_w // 2, bg_h // 2)]
selected_obj = objects[0]

dragging = False
offset_x = 0
offset_y = 0

# ----------------------------
# TRACKBAR CALLBACKS
# ----------------------------
def on_scale(val):
    """
    val: 1–100 (percent of max allowed scale)
    """
    if not selected_obj:
        return

    # Convert percent → scale ratio
    percent = max(val, 1) / 100.0
    scale = percent * fit_scale

    # Enforce minimum
    scale = max(scale, MIN_SCALE_RATIO)

    selected_obj.scale = scale


def on_brightness(val):
    if selected_obj:
        selected_obj.brightness = val - 100

def on_contrast(val):
    if selected_obj:
        selected_obj.contrast = val / 100.0

def on_blur(val):
    if selected_obj:
        selected_obj.blur = val if val % 2 == 1 else val + 1

# ----------------------------
# MOUSE CALLBACK
# ----------------------------
def mouse_cb(event, mx, my, flags, param):
    global selected_obj, dragging, offset_x, offset_y

    if event == cv2.EVENT_LBUTTONDOWN:
        selected_obj = None

        for obj in objects[::-1]:
            fw = int(orig_w * obj.scale)
            fh = int(orig_h * obj.scale)

            x0 = int(obj.x - fw // 2)
            y0 = int(obj.y - fh // 2)
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

                percent = int((obj.scale / fit_scale) * 100)
                cv2.setTrackbarPos("Scale %", "Editor", max(percent, 1))
                cv2.setTrackbarPos("Brightness", "Editor", obj.brightness + 100)
                cv2.setTrackbarPos("Contrast", "Editor", int(obj.contrast * 100))
                cv2.setTrackbarPos("Blur", "Editor", obj.blur)
                break

    elif event == cv2.EVENT_MOUSEMOVE and dragging and selected_obj:
        selected_obj.x = mx - offset_x
        selected_obj.y = my - offset_y

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

# ----------------------------
# WINDOW SETUP
# ----------------------------
cv2.namedWindow("Editor", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Editor", mouse_cb)

cv2.createTrackbar("Scale %", "Editor", 100, 100, on_scale)
cv2.createTrackbar("Brightness", "Editor", 100, 200, on_brightness)
cv2.createTrackbar("Contrast", "Editor", 100, 300, on_contrast)
cv2.createTrackbar("Blur", "Editor", 0, 25, on_blur)



# ---------------- SAVE SYSTEM ----------------
save_index = 1

def save_image(img):
    global save_index
    while True:
        filename = f"output_{save_index:04d}.jpg"
        path = os.path.join(OUTPUT_PATH, filename)
        if not os.path.exists(path):
            cv2.imwrite(path, img)
            print(f"✅ Saved {path}")
            save_index += 1
            break
        save_index += 1


# ----------------------------
# MAIN LOOP
# ----------------------------
while True:
    canvas = bg.copy()

    for obj in objects:

        fw = int(orig_w * obj.scale)
        fh = int(orig_h * obj.scale)
        # Clamp position so object stays inside frame
        obj.x = max(fw // 2, min(obj.x, bg_w - fw // 2))
        obj.y = max(fh // 2, min(obj.y, bg_h - fh // 2))

        if fw <= 0 or fh <= 0:
            continue

        fg_resized = cv2.resize(fg_rgb, (fw, fh), cv2.INTER_LANCZOS4)
        alpha_resized = cv2.resize(fg_alpha, (fw, fh), cv2.INTER_LINEAR)

        fg_adj = cv2.convertScaleAbs(
            fg_resized,
            alpha=obj.contrast,
            beta=obj.brightness
        )

        if obj.blur > 1:
            fg_adj = cv2.GaussianBlur(fg_adj, (obj.blur, obj.blur), 0)

        x = int(obj.x - fw // 2)
        y = int(obj.y - fh // 2)

        # -------- CLIPPING FIX --------
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(bg_w, x + fw)
        y1 = min(bg_h, y + fh)

        fx0 = x0 - x
        fy0 = y0 - y
        fx1 = fx0 + (x1 - x0)
        fy1 = fy0 + (y1 - y0)

        if x0 >= x1 or y0 >= y1:
            continue

        roi = canvas[y0:y1, x0:x1]
        alpha = (alpha_resized[fy0:fy1, fx0:fx1] / 255.0)[..., None]
        fg_part = fg_adj[fy0:fy1, fx0:fx1]

        canvas[y0:y1, x0:x1] = (
            alpha * fg_part + (1 - alpha) * roi
        ).astype(np.uint8)

        if obj.selected:
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 0), 2)

    cv2.imshow("Editor", canvas)

    key = cv2.waitKey(20) & 0xFF

    if key == 27:
        break
    elif key in (ord('d'), ord('D')):
        objects[:] = [o for o in objects if not o.selected]
    elif key in (ord('n'), ord('N')):
        objects.append(ObjectInstance(bg_w // 2, bg_h // 2))
    elif key in (ord('s'), ord('S')):
        save_image(canvas)

cv2.destroyAllWindows()
