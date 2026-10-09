"""Generate synthetic apple fixture images for visual grounding tests."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

_SEED = 42
_CANVAS_W = 800
_CANVAS_H = 600
_SCALE = 2  # render at 2x then downsample for anti-aliased edges
_W = _CANVAS_W * _SCALE
_H = _CANVAS_H * _SCALE

_WALL_TOP_COLOR = (220, 215, 208)
_WALL_BOT_COLOR = (200, 195, 185)
_TABLE_TOP_COLOR = (160, 118, 72)
_TABLE_EDGE_COLOR = (120, 85, 45)
_TABLE_Y_FRAC = 0.58  # table surface starts at this fraction of canvas height

_APPLE_R = 62 * _SCALE
_APPLE_COLOR = (200, 40, 30)
_STEM_COLOR = (100, 60, 20)
_LEAF_COLOR = (55, 140, 45)

_APPLE_POSITIONS_3 = [
    (200, 370),
    (400, 360),
    (620, 375),
]

_APPLE_POSITIONS_5 = [
    (200, 370),
    (400, 360),
    (620, 375),
    (300, 395),
    (520, 385),
]


def _draw_background(draw: ImageDraw.ImageDraw, w: int, h: int) -> None:
    table_y = int(h * _TABLE_Y_FRAC)
    for y in range(table_y):
        t = y / table_y
        r = int(_WALL_TOP_COLOR[0] + t * (_WALL_BOT_COLOR[0] - _WALL_TOP_COLOR[0]))
        g = int(_WALL_TOP_COLOR[1] + t * (_WALL_BOT_COLOR[1] - _WALL_TOP_COLOR[1]))
        b = int(_WALL_TOP_COLOR[2] + t * (_WALL_BOT_COLOR[2] - _WALL_TOP_COLOR[2]))
        draw.line([(0, y), (w, y)], fill=(r, g, b))

    edge_h = int(h * 0.04)
    draw.rectangle([0, table_y, w, table_y + edge_h], fill=_TABLE_EDGE_COLOR)

    for y in range(table_y + edge_h, h):
        grain = int(((y * 7 + (y % 13) * 3) % 20) - 10)
        r = max(0, min(255, _TABLE_TOP_COLOR[0] + grain))
        g = max(0, min(255, _TABLE_TOP_COLOR[1] + grain // 2))
        b = max(0, min(255, _TABLE_TOP_COLOR[2]))
        draw.line([(0, y), (w, y)], fill=(r, g, b))


def _draw_apple(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int) -> None:
    shadow_r = int(r * 1.15)
    shadow_img = Image.new("RGBA", (shadow_r * 2 + 20, shadow_r // 2 + 20), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow_img)
    sd.ellipse([10, 10, shadow_r * 2 + 10, shadow_r // 2 + 10], fill=(0, 0, 0, 60))
    shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=8))

    if hasattr(draw, "_image"):
        base = draw._image
        base.paste(shadow_img, (cx - shadow_r - 10, cy + r - shadow_r // 4), shadow_img)

    steps = 20
    for i in range(steps, 0, -1):
        frac = i / steps
        cr = int(r * frac)
        bright = 1.0 - 0.45 * (1 - frac)
        shade_r = max(0, min(255, int(_APPLE_COLOR[0] * bright)))
        shade_g = max(0, min(255, int(_APPLE_COLOR[1] * bright + 30 * (1 - frac))))
        shade_b = max(0, min(255, int(_APPLE_COLOR[2] * bright)))
        draw.ellipse(
            [cx - cr, cy - cr, cx + cr, cy + cr],
            fill=(shade_r, shade_g, shade_b),
        )

    hl_r = int(r * 0.3)
    hl_x = cx - int(r * 0.3)
    hl_y = cy - int(r * 0.3)
    for i in range(8, 0, -1):
        frac = i / 8
        alpha = int(200 * frac)
        cr = int(hl_r * frac)
        draw.ellipse(
            [hl_x - cr, hl_y - cr, hl_x + cr, hl_y + cr],
            fill=(255, 255, 255, alpha) if hasattr(draw, "_mode") else (255, 255, 255),
        )

    stem_x = cx
    stem_top_y = cy - r - int(r * 0.2)
    stem_bot_y = cy - r + int(r * 0.05)
    draw.line(
        [(stem_x, stem_top_y), (stem_x, stem_bot_y)],
        fill=_STEM_COLOR,
        width=max(2, int(r * 0.06)),
    )

    lx = stem_x + int(r * 0.1)
    ly = stem_top_y + int(r * 0.1)
    leaf_pts = [
        (lx, ly),
        (lx + int(r * 0.35), ly - int(r * 0.25)),
        (lx + int(r * 0.55), ly + int(r * 0.05)),
        (lx + int(r * 0.2), ly + int(r * 0.15)),
    ]
    draw.polygon(leaf_pts, fill=_LEAF_COLOR)


def _build_image(positions: list[tuple[int, int]]) -> Image.Image:
    img = Image.new("RGB", (_W, _H), _WALL_TOP_COLOR)
    draw = ImageDraw.Draw(img, "RGBA")
    draw._image = img

    _draw_background(draw, _W, _H)

    table_y = int(_H * _TABLE_Y_FRAC)
    for ox, oy in positions:
        cx = ox * _SCALE
        cy = (table_y + oy * _SCALE // _SCALE) if oy < _CANVAS_H else oy * _SCALE
        cy = int(_H * _TABLE_Y_FRAC) + int((oy - _CANVAS_H * _TABLE_Y_FRAC) * _SCALE)
        _draw_apple(draw, cx, cy, _APPLE_R)

    result = img.resize((_CANVAS_W, _CANVAS_H), Image.LANCZOS)
    return result


def _build_image_from_canvas_coords(positions: list[tuple[int, int]]) -> Image.Image:
    img = Image.new("RGB", (_W, _H), _WALL_TOP_COLOR)
    draw = ImageDraw.Draw(img, "RGBA")
    draw._image = img

    _draw_background(draw, _W, _H)

    for ox, oy in positions:
        cx = ox * _SCALE
        cy = oy * _SCALE
        _draw_apple(draw, cx, cy, _APPLE_R)

    result = img.resize((_CANVAS_W, _CANVAS_H), Image.LANCZOS)
    return result


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    fixtures_dir = project_root / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)

    table_y = int(_CANVAS_H * _TABLE_Y_FRAC)
    apple_surface_y = table_y + 55

    positions_3 = [
        (160, apple_surface_y),
        (400, apple_surface_y - 8),
        (640, apple_surface_y + 5),
    ]

    positions_5 = [
        (160, apple_surface_y),
        (400, apple_surface_y - 8),
        (640, apple_surface_y + 5),
        (280, apple_surface_y + 3),
        (520, apple_surface_y - 4),
    ]

    img_a = _build_image_from_canvas_coords(positions_3)
    img_b = _build_image_from_canvas_coords(positions_5)

    path_a = fixtures_dir / "image_a.jpg"
    path_b = fixtures_dir / "image_b.jpg"

    img_a.save(str(path_a), format="JPEG", quality=95, subsampling=0)
    img_b.save(str(path_b), format="JPEG", quality=95, subsampling=0)

    fixtures_config = {
        "question": "How many apples are on the table? Respond with only the number.",
        "image_a_path": "fixtures/image_a.jpg",
        "expected_a_contains": "3",
        "image_b_path": "fixtures/image_b.jpg",
        "expected_b_contains": "5",
    }

    config_path = fixtures_dir / "fixtures.json"
    config_path.write_text(json.dumps(fixtures_config, indent=2) + "\n", encoding="utf-8")

    print(f"image_a: {path_a} ({path_a.stat().st_size} bytes)")
    print(f"image_b: {path_b} ({path_b.stat().st_size} bytes)")
    print(f"fixtures.json: {config_path}")


if __name__ == "__main__":
    main()
