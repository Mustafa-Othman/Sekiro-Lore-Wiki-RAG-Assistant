"""
filter_boss_frames.py — pre-filter extracted frames down to likely boss-fight
screenshots, using Sekiro's on-screen boss health bar (orange/red bar across
the top of the screen, visible only during boss fights) as a detection signal.

This is a heuristic pre-filter, not a perfect classifier — it will still let
some false positives through (other enemies with health bars, weird lighting)
and may miss some genuine boss frames (bar obscured by an attack effect, or a
boss encountered by a runner in a way that renders the bar oddly). Treat its
output as a much smaller shortlist to manually review, not a final dataset.

    python filter_boss_frames.py --input frames/ --output likely_boss/
"""

import argparse
import os
import glob
import shutil
import cv2
import numpy as np

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")

# Sekiro's boss health bar is an orange-red gradient. These HSV ranges are a
# starting point — see the tuning note at the bottom of this file if your
# results look off (too many or too few detections).
HSV_LOWER = np.array([0, 80, 80])
HSV_UPPER = np.array([25, 255, 255])


def find_images(input_dir):
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(glob.glob(os.path.join(input_dir, "**", f"*{ext}"), recursive=True))
    return sorted(images)


def has_boss_bar(image_path, top_fraction, min_bar_fraction):
    img = cv2.imread(image_path)
    if img is None:
        return False, 0.0

    h, w = img.shape[:2]
    top_region = img[0: int(h * top_fraction), :]

    hsv = cv2.cvtColor(top_region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER) > 0

    # A real health bar is a thin, unbroken horizontal line — so check the
    # longest contiguous run WITHIN EACH ROW separately, and take the best
    # row. This rejects scattered textures (stone, foliage, torches) that
    # happen to fall in the same color range but never form one clean
    # unbroken line at any single row — unlike a naive "any match in this
    # column, anywhere in a tall region" check, which textures pass easily.
    best_run = 0
    for row in mask:
        run = 0
        for val in row:
            if val:
                run += 1
                best_run = max(best_run, run)
            else:
                run = 0

    bar_fraction = best_run / w
    return bar_fraction >= min_bar_fraction, bar_fraction


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Folder of frames to scan")
    parser.add_argument("--output", required=True, help="Folder for likely boss frames")
    parser.add_argument("--top-fraction", type=float, default=0.2,
                         help="Fraction of image height to scan, from the top")
    parser.add_argument("--min-bar-fraction", type=float, default=0.15,
                         help="Minimum width fraction the bar must span to count")
    parser.add_argument("--move", action="store_true",
                         help="Move files instead of copying")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    images = find_images(args.input)

    if not images:
        print(f"No images found in {args.input}")
        return

    print(f"Scanning {len(images)} images...")
    kept = 0
    for i, path in enumerate(images, 1):
        detected, score = has_boss_bar(path, args.top_fraction, args.min_bar_fraction)
        if detected:
            dest = os.path.join(args.output, os.path.basename(path))
            if args.move:
                shutil.move(path, dest)
            else:
                shutil.copy2(path, dest)
            kept += 1
        if i % 500 == 0:
            print(f"  ...{i}/{len(images)} scanned, {kept} kept so far")

    print(f"\nDone. Kept {kept}/{len(images)} images ({kept/len(images)*100:.1f}%) "
          f"-> {args.output}")
    print("This is a heuristic shortlist — manually skim the output folder next "
          "to remove false positives and confirm boss identity per image.")


if __name__ == "__main__":
    main()

# --- Tuning note ---
# If almost nothing gets detected: your HP bar's exact color may render
# differently after video compression — widen HSV_LOWER/HSV_UPPER (e.g. drop
# the saturation/value minimums) or lower --min-bar-fraction.
# If too much gets detected (false positives from torches, blood effects,
# orange lighting): narrow the HSV range or raise --min-bar-fraction and
# --top-fraction closer to where the bar actually sits in your cropped frames.
# Fastest way to tune: run on a small test folder first (a few hundred
# frames you already know contain boss fights) and adjust until the kept
# count roughly matches what you expect.
