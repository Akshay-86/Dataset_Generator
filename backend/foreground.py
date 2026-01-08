from rembg import remove
from PIL import Image
import numpy as np
import cv2


def remove_bg_and_autocrop(
    input_path: str,
    output_path: str,
    padding: int = 10
):
    # -------------------------------
    # 1. Remove background using rembg
    # -------------------------------
    input_image = Image.open(input_path).convert("RGBA")
    removed_bg = remove(input_image)

    # Convert PIL → NumPy (RGBA)
    img = np.array(removed_bg)

    # -------------------------------
    # 2. Extract alpha channel
    # -------------------------------
    alpha = img[:, :, 3]

    # Find non-transparent pixels
    ys, xs = np.where(alpha > 0)

    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("No foreground detected after background removal")

    # -------------------------------
    # 3. Bounding box + padding
    # -------------------------------
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    h, w = alpha.shape

    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w - 1, x_max + padding)
    y_max = min(h - 1, y_max + padding)

    # -------------------------------
    # 4. Crop
    # -------------------------------
    cropped = img[y_min:y_max + 1, x_min:x_max + 1]

    # -------------------------------
    # 5. Save final output
    # -------------------------------
    final_image = Image.fromarray(cropped, "RGBA")
    final_image.save(output_path)

    print(f"✅ Background removed, auto-cropped with {padding}px padding")
    print(f"📁 Saved to: {output_path}")


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    input_image_path = "assets/normalforeground/soldier.jpg"
    output_image_path = "assets/Foreground/soldier.png"

    remove_bg_and_autocrop(
        input_path=input_image_path,
        output_path=output_image_path,
        padding=10
    )
