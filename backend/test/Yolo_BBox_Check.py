import cv2
import os

def draw_yolo_boxes(image_path, label_path, output_path=None):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Image not found")

    h, w = img.shape[:2]

    # Read YOLO annotation file
    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id, x_center, y_center, bw, bh = map(float, parts)

        # Convert YOLO -> pixel coordinates
        x_center *= w
        y_center *= h
        bw *= w
        bh *= h

        x1 = int(x_center - bw / 2)
        y1 = int(y_center - bh / 2)
        x2 = int(x_center + bw / 2)
        y2 = int(y_center + bh / 2)

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"ID:{int(class_id)}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    cv2.namedWindow("YOLO Bounding Boxes", cv2.WINDOW_NORMAL)
    # Show image
    cv2.imshow("YOLO Bounding Boxes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save if output path given
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Saved output to: {output_path}")


# ---------------- USAGE ----------------
image_path = "assets/testimg/brightness+056_contrast082_blur07_img0029.jpg"
label_path = "assets/testimg/labels/brightness+056_contrast082_blur07_img0029.txt"
output_path = ""

draw_yolo_boxes(image_path, label_path, output_path)
