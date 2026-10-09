from __future__ import annotations

import base64
import io
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.exceptions import InvalidImageError

_MAX_PIXELS = 50_000_000  # reject decompression bombs before full decode

_ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
_FORMAT_TO_MIME = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


@dataclass(frozen=True)
class DetectedImage:
    format: str
    mime_type: str
    width: int
    height: int


@dataclass(frozen=True)
class PreparedImage:
    data: bytes
    mime_type: str
    base64: str
    width: int
    height: int
    resized: bool


def validate_image(data: bytes, declared_content_type: str | None) -> DetectedImage:
    if not data:
        raise InvalidImageError("uploaded file is empty")

    if (
        declared_content_type
        and declared_content_type != "application/octet-stream"
        and not declared_content_type.startswith("image/")
    ):
        raise InvalidImageError(
            f"declared content type {declared_content_type!r} is not an image type"
        )

    try:
        with Image.open(io.BytesIO(data)) as img:
            # Check pixel count from the header before loading all pixel data.
            width, height = img.size
            if width * height > _MAX_PIXELS:
                raise InvalidImageError("image exceeds pixel limit")
            img.verify()
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            fmt = img.format
            width, height = img.size
    except InvalidImageError:
        raise
    except Image.DecompressionBombError as exc:
        raise InvalidImageError("image exceeds pixel limit") from exc
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise InvalidImageError(f"image cannot be decoded: {exc}") from exc

    if fmt not in _ALLOWED_FORMATS:
        raise InvalidImageError(
            f"format {fmt!r} is not supported; allowed: JPEG, PNG, WebP"
        )

    return DetectedImage(
        format=fmt,
        mime_type=_FORMAT_TO_MIME[fmt],
        width=width,
        height=height,
    )


def prepare_for_model(
    data: bytes, detected: DetectedImage, max_dimension: int
) -> PreparedImage:
    with Image.open(io.BytesIO(data)) as img:
        img = ImageOps.exif_transpose(img)
        width, height = img.size

        if width <= max_dimension and height <= max_dimension:
            raw = data
            resized = False
            out_width, out_height = width, height
        else:
            scale = max_dimension / max(width, height)
            new_w = round(width * scale)
            new_h = round(height * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            buf = io.BytesIO()

            fmt = detected.format
            if fmt == "JPEG":
                img = img.convert("RGB")
                img.save(buf, format="JPEG", quality=90, optimize=True)
            elif fmt == "PNG":
                img.save(buf, format="PNG", optimize=True)
            else:
                img.save(buf, format="WEBP", quality=90)

            raw = buf.getvalue()
            resized = True
            out_width, out_height = new_w, new_h

    b64 = base64.b64encode(raw).decode("ascii")
    return PreparedImage(
        data=raw,
        mime_type=detected.mime_type,
        base64=b64,
        width=out_width,
        height=out_height,
        resized=resized,
    )
