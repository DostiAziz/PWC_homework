from operator import add
from typing import Annotated, Literal, TypedDict

from pwc_support.domain.models import CancellationPreview, Citation, Task, TaskResult, TraceEvent


class SupportState(TypedDict, total=False):
    message: str
    customer_id: str
    pending_cancellation: CancellationPreview | None
    awaiting_cancel: bool
    confirmation: Literal["yes", "no", "unclear"] | None
    task: Task
    tasks: tuple[Task, ...]
    results: Annotated[list[TaskResult], add]
    ordered_results: tuple[TaskResult, ...]
    direct_response: str
    response: str
    citations: tuple[Citation, ...]
    events: Annotated[list[TraceEvent], add]
