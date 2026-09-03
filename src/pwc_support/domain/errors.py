from enum import StrEnum


class ErrorCode(StrEnum):
    INVALID_INPUT = "INVALID_INPUT"
    UNSUPPORTED_LANGUAGE = "UNSUPPORTED_LANGUAGE"
    DUPLICATE_MESSAGE = "DUPLICATE_MESSAGE"
    OLLAMA_UNAVAILABLE = "OLLAMA_UNAVAILABLE"
    MODEL_TIMEOUT = "MODEL_TIMEOUT"
    MODEL_OUTPUT_INVALID = "MODEL_OUTPUT_INVALID"
    EMBEDDING_FAILED = "EMBEDDING_FAILED"
    RETRIEVAL_FAILED = "RETRIEVAL_FAILED"
    EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"
    POLICY_REVIEW_REQUIRED = "POLICY_REVIEW_REQUIRED"
    CASE_ACCESS_DENIED = "CASE_ACCESS_DENIED"
    VERSION_CONFLICT = "VERSION_CONFLICT"
    CHECKPOINT_FAILED = "CHECKPOINT_FAILED"
    DELIVERY_FAILED = "DELIVERY_FAILED"
    RISK_CLASSIFICATION_UNAVAILABLE = "RISK_CLASSIFICATION_UNAVAILABLE"
    STALE_INBOUND_CLAIM = "STALE_INBOUND_CLAIM"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"


class SupportError(RuntimeError):
    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class RiskClassificationUnavailable(SupportError):
    """The semantic risk model could not produce a safe routing decision."""

    def __init__(self, *, failure_class: str, message: str) -> None:
        super().__init__(ErrorCode.RISK_CLASSIFICATION_UNAVAILABLE, message)
        self.failure_class = failure_class
