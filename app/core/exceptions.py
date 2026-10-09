from __future__ import annotations


class VQAError(Exception):
    status_code: int = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidRequestError(VQAError):
    status_code = 400


class PayloadTooLargeError(VQAError):
    status_code = 413


class InvalidImageError(VQAError):
    status_code = 400


class ProviderNotConfiguredError(VQAError):
    status_code = 503


class UpstreamTimeoutError(VQAError):
    status_code = 504


class UpstreamError(VQAError):
    status_code = 502


class UpstreamRateLimitedError(VQAError):
    status_code = 503


class AuditWriteError(VQAError):
    status_code = 500
