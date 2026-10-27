"""Structured tool events out of a running DSPy program.

DSPy already has the two hooks an AG-UI bridge needs:

* ``dspy.streamify`` runs the program with a ``send_stream`` in settings and
  yields everything that lands on it. Unknown objects pass through untouched
  ("wildcard case ... customized streaming behavior").
* ``BaseCallback.on_tool_start/end`` fire around every ``dspy.Tool`` call with
  a ``call_id``, the tool instance and the kwargs.

So the bridge is a callback that drops small dataclasses onto the stream, and
an adapter that turns those plus the text chunks into AG-UI events. No
subclassing of ReAct, no string parsing of status messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dspy.dsp.utils.settings import settings
from dspy.streaming.messages import sync_send_to_stream
from dspy.utils.callback import BaseCallback


@dataclass
class ToolStart:
    call_id: str
    name: str
    args: dict[str, Any]


@dataclass
class ToolEnd:
    call_id: str
    output: Any
    error: BaseException | None


class FrontendToolCall(BaseException):
    """Raised by a frontend tool stub to hand the call to the client.

    BaseException on purpose: ReAct swallows ``Exception`` into the trajectory
    as an "Execution error" observation, which is the opposite of what we want.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name


class AgUiCallback(BaseCallback):
    """Pushes tool starts/ends onto the streamify stream as structured objects."""

    skip = {"finish"}  # ReAct's internal terminator is not a real tool

    def on_tool_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        stream = settings.send_stream
        if stream is None or instance.name in self.skip:
            return
        sync_send_to_stream(stream, ToolStart(call_id, instance.name, _kwargs(inputs)))

    def on_tool_end(self, call_id: str, outputs: Any, exception: BaseException | None = None):
        stream = settings.send_stream
        if stream is None or outputs == "Completed." or isinstance(exception, FrontendToolCall):
            return
        sync_send_to_stream(stream, ToolEnd(call_id, outputs, exception))


def _kwargs(inputs: dict[str, Any]) -> dict[str, Any]:
    # Tool.__call__(self, *args, **kwargs): with_callbacks records them under these keys.
    if set(inputs) <= {"args", "kwargs"}:
        return dict(inputs.get("kwargs") or {})
    return inputs
