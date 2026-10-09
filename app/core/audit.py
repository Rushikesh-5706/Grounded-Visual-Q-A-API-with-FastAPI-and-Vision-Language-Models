from __future__ import annotations

import hashlib
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

from app.core.exceptions import AuditWriteError

_lock = threading.Lock()


def audit_image_payload(
    request_id: str,
    image_bytes: bytes,
    audit_path: Path | None = None,
) -> str:
    if audit_path is None:
        from app.core.config import get_settings

        audit_path = get_settings().audit_log_path

    digest = hashlib.sha256(image_bytes).hexdigest()
    timestamp = datetime.now(UTC).isoformat(timespec="milliseconds")
    line = f"{timestamp} | {request_id} | {digest}\n"

    with _lock:
        try:
            with audit_path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
        except OSError as exc:
            raise AuditWriteError(f"cannot write audit log: {exc}") from exc

    return digest
