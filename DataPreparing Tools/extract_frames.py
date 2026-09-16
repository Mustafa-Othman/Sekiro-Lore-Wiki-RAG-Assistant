"""
extract_frames.py — pull frames from gameplay video(s) for YOLO dataset building.

Usage (single video):
    python extract_frames.py --video walkthrough.mp4 --out frames/ --interval 1.0

Usage (whole folder of videos):
    python extract_frames.py --video-dir ./videos/ --out frames/ --interval 1.0

    --video      path to a single input video file
    --video-dir  path to a folder of video files — processes every video inside it
                 (recursively), saving each video's frames into its own subfolder under --out
    --out        output folder for extracted frames (created if missing)
    --interval   seconds between extracted frames (default: 1.0)
    --start      start time in seconds, e.g. skip an intro (default: 0)
    --end        end time in seconds, e.g. stop before credits (default: full video)
    --dedupe     skip near-identical consecutive frames (default: on)
    --blur-thresh  minimum sharpness score to keep a frame; raises this to be
                   stricter about dropping blurry/motion-blurred frames (default: 60)
    --crop       crop the gameplay region out of overlays/UI before saving, as
                 "x,y,width,height" in pixels (top-left corner + box size).
                 Find these values by opening one extracted frame in an image
                 viewer that shows pixel coordinates. Example: --crop 310,140,715,410

Requires: opencv-python, numpy
    pip install opencv-python numpy
"""

import argparse
import os
import glob
import cv2
import numpy as np

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".m4v")


def parse_crop(crop_str):
    """Parse '--crop x,y,width,height' into a tuple of ints, or None if not given."""
    if not crop_str:
        return None
    try:
        x, y, w, h = (int(v.strip()) for v in crop_str.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--crop must be 'x,y,width,height' e.g. --crop 310,140,715,410"
        )
    return x, y, w, h


def apply_crop(frame, crop):
    """Crop a frame to the gameplay region, clamped to the frame's actual bounds."""
    if crop is None:
        return frame
    x, y, w, h = crop
    frame_h, frame_w = frame.shape[:2]
    x2 = min(x + w, frame_w)
    y2 = min(y + h, frame_h)
    return frame[y:y2, x:x2]


def is_too_blurry(frame, threshold: float) -> bool:
    """Laplacian variance is a common cheap sharpness proxy — lower = blurrier."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


def is_near_duplicate(frame, prev_frame, threshold: float = 8.0) -> bool:
    """Cheap frame-diff check to skip frames that barely changed (e.g. paused/static)."""
    if prev_frame is None:
        return False
    diff = cv2.absdiff(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
    )
    return float(np.mean(diff)) < threshold


def extract_frames(video_path, out_dir, interval, start, end, dedupe, blur_thresh, crop=None):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    end = end if end is not None else duration

    frame_interval = max(1, int(round(fps * interval)))

    print(f"Video: {video_path}")
    print(f"  fps={fps:.2f}, duration={duration:.1f}s, sampling every {interval}s "
          f"(~{frame_interval} frames)")
    if crop:
        print(f"  cropping to x={crop[0]}, y={crop[1]}, width={crop[2]}, height={crop[3]}")

    saved, skipped_blur, skipped_dupe, frame_idx = 0, 0, 0, 0
    prev_saved_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        if timestamp < start:
            frame_idx += 1
            continue
        if timestamp > end:
            break

        if frame_idx % frame_interval == 0:
            frame = apply_crop(frame, crop)
            if is_too_blurry(frame, blur_thresh):
                skipped_blur += 1
            elif dedupe and is_near_duplicate(frame, prev_saved_frame):
                skipped_dupe += 1
            else:
                filename = os.path.join(out_dir, f"frame_{timestamp:07.1f}s.png")
                cv2.imwrite(filename, frame)
                prev_saved_frame = frame
                saved += 1

        frame_idx += 1

    cap.release()
    print(f"Done. Saved {saved} frames -> {out_dir}")
    print(f"  Skipped {skipped_blur} blurry frames, {skipped_dupe} near-duplicate frames")


def find_videos(video_dir):
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(glob.glob(os.path.join(video_dir, "**", f"*{ext}"), recursive=True))
    return sorted(videos)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", help="Path to a single input video")
    group.add_argument("--video-dir", help="Path to a folder of videos to process")
    parser.add_argument("--out", required=True, help="Output folder for frames")
    parser.add_argument("--interval", type=float, default=1.0,
                         help="Seconds between extracted frames")
    parser.add_argument("--start", type=float, default=0.0, help="Start time (seconds)")
    parser.add_argument("--end", type=float, default=None, help="End time (seconds)")
    parser.add_argument("--no-dedupe", dest="dedupe", action="store_false",
                         help="Disable near-duplicate frame skipping")
    parser.add_argument("--blur-thresh", type=float, default=60.0,
                         help="Minimum sharpness score to keep a frame")
    parser.add_argument("--crop", type=parse_crop, default=None,
                         help="Crop region as 'x,y,width,height' in pixels, "
                              "e.g. --crop 310,140,715,410")
    args = parser.parse_args()

    if args.video:
        extract_frames(
            video_path=args.video,
            out_dir=args.out,
            interval=args.interval,
            start=args.start,
            end=args.end,
            dedupe=args.dedupe,
            blur_thresh=args.blur_thresh,
            crop=args.crop,
        )
    else:
        videos = find_videos(args.video_dir)
        if not videos:
            print(f"No video files found in {args.video_dir} "
                  f"(looked for {', '.join(VIDEO_EXTENSIONS)})")
        else:
            print(f"Found {len(videos)} video(s) in {args.video_dir}\n")
            for video_path in videos:
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                video_out_dir = os.path.join(args.out, video_name)
                try:
                    extract_frames(
                        video_path=video_path,
                        out_dir=video_out_dir,
                        interval=args.interval,
                        start=args.start,
                        end=args.end,
                        dedupe=args.dedupe,
                        blur_thresh=args.blur_thresh,
                        crop=args.crop,
                    )
                except Exception as e:
                    print(f"  Skipped {video_path} due to error: {e}")
                print()