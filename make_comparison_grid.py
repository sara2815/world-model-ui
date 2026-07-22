"""
make_comparison_grid.py

Combines the 4 prompt_N/pred_after.png outputs from test_prompt_variants.py into
a single grid image, with each panel's prompt text displayed above it. Also
includes the shared input image as the first panel for reference.

Run this LOCALLY (no GPU needed) after downloading prompt_results/ from Fir.

Usage:
  python make_comparison_grid.py --results_dir "C:\\Users\\adyes\\Downloads\\prompt_results" --output comparison.png
"""

import argparse
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Common locations for a reasonably legible TrueType font across platforms.
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int, draw: ImageDraw.ImageDraw) -> str:
    """Wrap text to fit max_width pixels, using the given font."""
    words = text.split()
    lines, current = [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


def make_panel(image_path: Path, caption: str, panel_w: int, panel_h: int,
               caption_h: int, font: ImageFont.FreeTypeFont) -> Image.Image:
    panel = Image.new("RGB", (panel_w, panel_h + caption_h), "white")
    draw = ImageDraw.Draw(panel)

    # Draw wrapped caption text at the top
    wrapped = wrap_text(caption, font, panel_w - 16, draw)
    draw.multiline_text((8, 8), wrapped, fill="black", font=font, spacing=4)

    # Paste the image below the caption, scaled to fit panel_w x panel_h
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((panel_w, panel_h), Image.LANCZOS)
    x_off = (panel_w - img.width) // 2
    panel.paste(img, (x_off, caption_h))

    # Border for clarity between panels
    draw.rectangle([0, 0, panel_w - 1, panel_h + caption_h - 1], outline="gray", width=1)
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True, type=Path,
                     help="Path to the downloaded prompt_results folder (contains prompt_1..4, input.png).")
    ap.add_argument("--output", default="comparison.png", type=Path)
    ap.add_argument("--panel_width", type=int, default=420)
    ap.add_argument("--panel_height", type=int, default=420)
    ap.add_argument("--caption_height", type=int, default=140,
                     help="Space reserved above each image for wrapped prompt text.")
    ap.add_argument("--columns", type=int, default=2, help="Grid columns (2 -> 2x2 for 4 panels, +input).")
    args = ap.parse_args()

    font = load_font(16)

    # Collect panels: input first, then prompt_1..4
    panels_info = []
    input_path = args.results_dir / "input.png"
    if input_path.is_file():
        panels_info.append((input_path, "Input image"))

    for i in range(1, 5):
        pred_path = args.results_dir / f"prompt_{i}" / "pred_after.png"
        prompt_txt_path = args.results_dir / f"prompt_{i}" / "prompt_used.txt"
        if not pred_path.is_file():
            print(f"[skip] missing {pred_path}")
            continue
        caption = f"[{i}] " + (prompt_txt_path.read_text(encoding="utf-8").strip()
                                if prompt_txt_path.is_file() else "(prompt text not found)")
        panels_info.append((pred_path, caption))

    if not panels_info:
        raise RuntimeError(f"No images found under {args.results_dir}")

    # Compute the caption height needed for the LONGEST wrapped caption, so no
    # panel's text overlaps its image regardless of prompt length.
    measure_img = Image.new("RGB", (10, 10))
    measure_draw = ImageDraw.Draw(measure_img)
    line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + 4
    max_lines = 1
    for _, caption in panels_info:
        wrapped = wrap_text(caption, font, args.panel_width - 16, measure_draw)
        max_lines = max(max_lines, wrapped.count("\n") + 1)
    caption_h = max(args.caption_height, 16 + max_lines * line_height)

    n = len(panels_info)
    cols = args.columns
    rows = (n + cols - 1) // cols

    panel_total_h = args.panel_height + caption_h
    grid = Image.new("RGB", (cols * args.panel_width, rows * panel_total_h), "white")

    for idx, (img_path, caption) in enumerate(panels_info):
        panel = make_panel(img_path, caption, args.panel_width, args.panel_height,
                            caption_h, font)
        r, c = divmod(idx, cols)
        grid.paste(panel, (c * args.panel_width, r * panel_total_h))

    grid.save(args.output)
    print(f"[done] saved comparison grid -> {args.output.resolve()}")


if __name__ == "__main__":
    main()