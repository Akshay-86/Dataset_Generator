import cv2
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse


# ------------------ UTILS ------------------

def is_url(path: str) -> bool:
    return urlparse(path).scheme in ("http", "https")


def download_video(url: str, output_dir: str) -> str:
    """
    Download video using yt-dlp and return downloaded file path.
    """
    os.makedirs(output_dir, exist_ok=True)

    output_template = os.path.join(output_dir, "video.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", "bestvideo+bestaudio/best",
        "--merge-output-format", "mp4",
        "-o", output_template,"--force-overwrites",
        url
    ]

    subprocess.run(cmd, check=True)

    for file in Path(output_dir).iterdir():
        if file.suffix in (".mp4", ".mkv", ".webm"):
            return str(file)

    raise FileNotFoundError("Downloaded video not found")


# ------------------ FRAME EXTRACTION ------------------

def extract_frames(
    video_path: str,
    output_dir: str,
    mode: str = "seconds",
    value: float = 1.0
):
    """
    Extract frames from video.

    mode:
        "seconds" → 1 frame every `value` seconds
        "fps"     → `value` frames per second
    """

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] FPS: {fps}")
    print(f"[INFO] Total Frames: {total_frames}")

    if mode == "seconds":
        frame_interval = int(fps * value)
    elif mode == "fps":
        frame_interval = max(1, int(fps / value))
    else:
        raise ValueError("mode must be 'seconds' or 'fps'")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = f"frame_{saved_count:06d}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"[DONE] Extracted {saved_count} frames")


# ------------------ MAIN PIPELINE ------------------

def process_video(
    input_source: str,
    frames_dir: str,
    mode: str = "seconds",
    value: float = 1.0,
    download_dir: str = "assets/videos"
):
    """
    Unified pipeline:
    - If URL → download
    - If file → use directly
    - Extract frames
    """

    if is_url(input_source):
        print("[INFO] URL detected, downloading video...")
        video_path = download_video(input_source, download_dir)
    else:
        if not os.path.isfile(input_source):
            raise FileNotFoundError("Local video file not found")
        print("[INFO] Local video detected")
        video_path = input_source

    extract_frames(
        video_path=video_path,
        output_dir=frames_dir,
        mode=mode,
        value=value
    )


# ------------------ EXAMPLE ------------------

if __name__ == "__main__":

    INPUT = "assets/videos/sea.mp4"
    # OR
    # INPUT = "/home/user/videos/sample.mp4"

    process_video(
        input_source=INPUT,
        frames_dir="assets/testimg",
             # "seconds" or "fps"
        value=10        # 10 sec OR 10 fps
    )
