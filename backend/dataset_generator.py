import cv2
import json
import os
import random
import shutil


try:
    with open("backend/data.json","r") as f:
        data = json.load(f)
except FileNotFoundError:
    print("data.json not found.")
except json.JSONDecodeError:
    print("Error decoding data.json.")

BASE_DIR = data["base_dir"]
OUT_DIR = data["out_dir"]
COUNT = data["count"]

GLOBAL_RANDOM = {
    "brightness": tuple(data["global_random"]["brightness"]),
    "contrast": tuple(data["global_random"]["contrast"]),
    "blur": tuple(data["global_random"]["blur"])
}

def apply_filters(img, f):
    out = cv2.convertScaleAbs(img, alpha=f["contrast"], beta=f["brightness"])
    if f["blur"] > 1:
        k = f["blur"] if f["blur"] % 2 else f["blur"] + 1
        out = cv2.GaussianBlur(out, (k, k), 0)
    return out

def process(scene):
    img = cv2.imread(f"{BASE_DIR}/{scene}.png")

    with open(f"{BASE_DIR}/json/{scene}.json") as f:
        meta = json.load(f)

    yolo_src = f"{BASE_DIR}/yolo_label/{scene}.txt"

    out_base = f"{OUT_DIR}/{scene}"
    os.makedirs(out_base + "/images", exist_ok=True)
    os.makedirs(out_base + "/json", exist_ok=True)
    os.makedirs(out_base + "/yolo_label", exist_ok=True)

    for i in range(COUNT):
        gf = {
            "brightness": random.randint(*GLOBAL_RANDOM["brightness"]),
            "contrast": random.uniform(*GLOBAL_RANDOM["contrast"]),
            "blur": random.randint(*GLOBAL_RANDOM["blur"])
        }

        final = apply_filters(img, gf)
        name = f"{scene}_{i:04d}"

        cv2.imwrite(f"{out_base}/images/{name}.png", final)
        shutil.copy(yolo_src, f"{out_base}/yolo_label/{name}.txt")

        meta_copy = meta.copy()
        meta_copy["global_filters"] = gf

        with open(f"{out_base}/json/{name}.json", "w") as f:
            json.dump(meta_copy, f, indent=4)

        print("Generated:", name)

if __name__ == "__main__":
    for f in os.listdir(BASE_DIR):
        if f.endswith(".png"):
            process(f.replace(".png", ""))
