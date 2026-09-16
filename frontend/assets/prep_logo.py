"""Prep header logo from sekiro_wallpaper (face crop, light desaturation)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance

SRC = Path(
    r"C:\Users\drsha\.cursor\projects\d-Mustafa-programming\assets"
    r"\c__Users_drsha_AppData_Roaming_Cursor_User_workspaceStorage_"
    r"65bbff354ce4a3ba20545d5efcdd1d2e_images_sekiro_wallpaper-"
    r"09e809cd-76f6-442f-86aa-993563d7b4e8.png"
)
OUT = Path(__file__).resolve().parent / "logo_header.png"


def main() -> None:
    im = Image.open(SRC).convert("RGB")
    w, h = im.size
    # Face / topknot sit in the upper-right with the red sun behind
    cx, cy = int(w * 0.62), int(h * 0.22)
    side = int(min(w, h) * 0.42)
    left = max(0, cx - side // 2)
    top = max(0, cy - side // 2)
    right = min(w, left + side)
    bottom = min(h, top + side)
    crop = im.crop((left, top, right, bottom))

    # Slight desaturation (~18%) — keep cream/gold warmth
    graded = ImageEnhance.Color(crop).enhance(0.82)
    logo = graded.resize((160, 160), Image.Resampling.LANCZOS)

    # Soft rounded-square mask (~14px radius at 160px ≈ site 52px mark)
    radius = 28
    mask = Image.new("L", (160, 160), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, 159, 159), radius=radius, fill=255)
    rgba = logo.convert("RGBA")
    rgba.putalpha(mask)
    rgba.save(OUT, "PNG")
    print("logo crop", (left, top, right, bottom), "->", OUT)


if __name__ == "__main__":
    main()
