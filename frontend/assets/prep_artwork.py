"""One-shot asset prep: avatar crop + welcome backdrop from Sekiro artwork."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter

SRC = Path(
    r"C:\Users\drsha\.cursor\projects\d-Mustafa-programming\assets"
    r"\c__Users_drsha_AppData_Roaming_Cursor_User_workspaceStorage_"
    r"65bbff354ce4a3ba20545d5efcdd1d2e_images_devlinart_sekiro-"
    r"aca8667c-acd7-4903-9685-d178e5c17c9c.png"
)
OUT = Path(__file__).resolve().parent
INDIGO = (0x14, 0x17, 0x1F)


def grade(im: Image.Image) -> Image.Image:
    """Desaturate ~40%+, then grade toward site indigo (#14171F)."""
    rgb = im.convert("RGB")
    # Stronger pullback on the warm reds so the art sits in the UI palette
    desat = ImageEnhance.Color(rgb).enhance(0.45)
    overlay = Image.new("RGB", desat.size, INDIGO)
    multiplied = ImageChops.multiply(desat, overlay)
    # color-burn-ish: darken toward indigo where art is midtone
    burned = ImageChops.darker(multiplied, Image.blend(desat, overlay, 0.35))
    stepped = Image.blend(desat, burned, 0.65)
    return Image.blend(stepped, overlay, 0.22)


def main() -> None:
    im = Image.open(SRC).convert("RGB")
    w, h = im.size

    # Tighter head / shoulders / blade tip — cut maple field as much as possible
    cx, cy = int(w * 0.52), int(h * 0.24)
    side = int(min(w, h) * 0.48)
    left = max(0, cx - side // 2)
    top = max(0, cy - side // 2)
    right = min(w, left + side)
    bottom = min(h, top + side)
    crop = im.crop((left, top, right, bottom))

    avatar = grade(crop).resize((160, 160), Image.Resampling.LANCZOS)
    mask = Image.new("L", (160, 160), 0)
    ImageDraw.Draw(mask).ellipse((1, 1, 158, 158), fill=255)
    avatar_rgba = avatar.convert("RGBA")
    avatar_rgba.putalpha(mask)
    avatar_rgba.save(OUT / "avatar_assistant.png", "PNG")

    # Welcome backdrop: graded full art, softer vignette, ~28% peak opacity
    full = grade(im)
    max_w = 900
    scale = max_w / full.width
    full = full.resize((max_w, int(full.height * scale)), Image.Resampling.LANCZOS)
    fw, fh = full.size
    rgba = full.convert("RGBA")

    # Peak alpha ~72/255 (~28%). Soft edge falloff starts late so the figure stays readable.
    vignette = Image.new("L", (fw, fh), 0)
    draw = ImageDraw.Draw(vignette)
    pad_x, pad_y = int(fw * 0.02), int(fh * 0.02)
    draw.ellipse((-pad_x, -pad_y, fw + pad_x, fh + pad_y), fill=72)
    vignette = vignette.filter(ImageFilter.GaussianBlur(radius=int(min(fw, fh) * 0.10)))

    edge = Image.new("L", (fw, fh), 255)
    ed = ImageDraw.Draw(edge)
    m = int(min(fw, fh) * 0.06)
    # Only fade the outer rim — keep the center figure intact
    border = Image.new("L", (fw, fh), 0)
    bd = ImageDraw.Draw(border)
    bd.rectangle((0, 0, fw - 1, fh - 1), fill=255)
    bd.rectangle((m, m, fw - m - 1, fh - m - 1), fill=0)
    border = border.filter(ImageFilter.GaussianBlur(radius=m * 2))
    # Invert border fade into a keep-mask
    keep = ImageChops.invert(border)
    rgba.putalpha(ImageChops.multiply(vignette, keep))
    rgba.save(OUT / "welcome_backdrop.png", "PNG")
    print("avatar crop", (left, top, right, bottom))
    print("wrote", OUT / "avatar_assistant.png")
    print("wrote", OUT / "welcome_backdrop.png", rgba.size)


if __name__ == "__main__":
    main()
