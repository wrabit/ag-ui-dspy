"""RunAgentInput in, AG-UI events out, for any dspy.Module."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import dspy
from ag_ui.core import (
    AssistantMessage,
    BaseEvent,
    EventType as T,
    Message,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StepStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    Tool,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
    ToolMessage,
    UserMessage,
)
from dspy.streaming import StatusMessage, StatusMessageProvider, StreamListener, StreamResponse

from .callback import AgUiCallback, FrontendToolCall, ToolEnd, ToolStart


class _Quiet(StatusMessageProvider):
    """Tool events come through AgUiCallback as structured objects, not strings."""

    def tool_start_status_message(self, instance, inputs):
        return None

    def tool_end_status_message(self, outputs):
        return None


class DSPyAgent:
    """Wrap a dspy.Module so it speaks AG-UI.

    program_factory(tools) -> dspy.Module. It gets the frontend tools from the
    request (as dspy.Tool stubs) so a ReAct agent can offer them to the LM
    alongside its own backend tools.

    input_field / output_field: the signature fields that carry the user's
    message and the reply. history_field: a ``dspy.History`` input to receive
    the prior turns, or None to send only the latest message.
    """

    def __init__(
        self,
        program_factory,
        *,
        input_field: str = "question",
        output_field: str = "answer",
        history_field: str | None = "history",
        name: str = "dspy",
    ):
        self.program_factory = program_factory
        self.input_field = input_field
        self.output_field = output_field
        self.history_field = history_field
        self.name = name

    # ------------------------------------------------------------------ run
    async def run(self, run_input: RunAgentInput) -> AsyncIterator[BaseEvent]:
        thread_id, run_id = run_input.thread_id, run_input.run_id
        yield RunStartedEvent(type=T.RUN_STARTED, thread_id=thread_id, run_id=run_id)

        program = self.program_factory(_frontend_tools(run_input.tools))
        # streamify snapshots settings.callbacks when called, so register ours first.
        # Stream the reply token by token only when exactly one predictor generates the output
        # field. ReActV2 returns it through its submit tool (no predictor has it); a module where
        # several predictors share the field, or post-processes it, gets the final value whole.
        owners = [p for _, p in program.named_predictors() if self.output_field in p.signature.output_fields]
        listeners = [StreamListener(signature_field_name=self.output_field)] if len(owners) == 1 else []
        with dspy.context(callbacks=[*dspy.settings.callbacks, AgUiCallback()]):
            streamer = dspy.streamify(
                program,
                status_message_provider=_Quiet(),
                stream_listeners=listeners,
                include_final_prediction_in_output_stream=True,
            )
        kwargs = self._inputs(run_input.messages, react_v2=isinstance(program, dspy.ReActV2))
        message_id = f"{run_id}-reply"
        started = False
        streamed = ""
        open_calls: dict[str, str] = {}

        def start_text():
            nonlocal started
            started = True
            return TextMessageStartEvent(type=T.TEXT_MESSAGE_START, message_id=message_id, role="assistant")

        try:
            async for item in streamer(**kwargs):
                if isinstance(item, ToolStart):
                    open_calls[item.call_id] = item.name
                    yield ToolCallStartEvent(type=T.TOOL_CALL_START, tool_call_id=item.call_id, tool_call_name=item.name)
                    yield ToolCallArgsEvent(type=T.TOOL_CALL_ARGS, tool_call_id=item.call_id, delta=json.dumps(item.args, default=str))
                    yield ToolCallEndEvent(type=T.TOOL_CALL_END, tool_call_id=item.call_id)
                elif isinstance(item, ToolEnd):
                    if item.call_id not in open_calls:
                        continue
                    open_calls.pop(item.call_id)
                    content = _text(item.error) if item.error else _text(item.output)
                    yield ToolCallResultEvent(
                        type=T.TOOL_CALL_RESULT, message_id=f"{item.call_id}-result",
                        tool_call_id=item.call_id, content=content, role="tool",
                    )
                elif isinstance(item, StreamResponse):
                    if not item.chunk:  # the listener's final flush can be empty; AG-UI rejects empty deltas
                        continue
                    if not started:
                        yield start_text()
                    streamed += item.chunk
                    yield TextMessageContentEvent(type=T.TEXT_MESSAGE_CONTENT, message_id=message_id, delta=item.chunk)
                elif isinstance(item, StatusMessage):
                    yield StepStartedEvent(type=T.STEP_STARTED, step_name=item.message)
                elif isinstance(item, dspy.Prediction):
                    final = _text(getattr(item, self.output_field, ""))
                    if not started:
                        yield start_text()
                    if final and not streamed:  # cache hit or LM without streaming
                        yield TextMessageContentEvent(type=T.TEXT_MESSAGE_CONTENT, message_id=message_id, delta=final)
                    yield TextMessageEndEvent(type=T.TEXT_MESSAGE_END, message_id=message_id)
        except FrontendToolCall:
            # The TOOL_CALL_* events already went out via the callback. The run
            # ends here with no result; the client executes the tool and starts
            # a new run whose messages carry the ToolMessage.
            pass
        except Exception as exc:  # noqa: BLE001
            yield RunErrorEvent(type=T.RUN_ERROR, message=str(exc))
            return
        yield RunFinishedEvent(type=T.RUN_FINISHED, thread_id=thread_id, run_id=run_id)

    # ------------------------------------------------------------- messages
    def _inputs(self, messages: list[Message], react_v2: bool = False) -> dict[str, Any]:
        """Map the AG-UI message list onto the module's inputs.

        * Completed turns (user -> assistant, with any tool exchange) become
          ``dspy.History`` entries of ``{input_field, output_field}``. For a
          ``dspy.ReActV2`` program they are shaped like its own history events
          (a ``submit`` tool call carrying the answer), since its predictor
          only renders ``next_thought`` / ``tool_calls`` for past turns.
        * The last user message is the question.
        * Tool results that follow it (a frontend tool the client has just
          executed) are appended to the question, so the next ReAct trajectory
          sees the answer up front and does not ask again.
        """
        turns: list[dict[str, str]] = []
        pending_q: str | None = None
        last_user: str | None = None
        calls: dict[str, str] = {}  # tool_call_id -> "name(args)"
        trailing: list[str] = []  # tool results after the last user message
        for m in messages:
            if isinstance(m, UserMessage):
                pending_q = _text(m.content)
                last_user = pending_q
                trailing = []
            elif isinstance(m, AssistantMessage):
                a = _text(m.content) or ""
                for c in m.tool_calls or []:
                    calls[c.id] = f"{c.function.name}({c.function.arguments})"
                    a += f"\n[called {calls[c.id]}]"
                if pending_q is not None:
                    turns.append({self.input_field: pending_q, self.output_field: a})
                    pending_q = None
                elif turns:
                    turns[-1][self.output_field] += "\n" + a
            elif isinstance(m, ToolMessage):
                call = calls.get(m.tool_call_id, m.tool_call_id)
                trailing.append(f"[You already called {call}; the result was: {_text(m.content)}. Do not call it again.]")
                if turns:
                    turns[-1][self.output_field] += f"\n[tool result {m.tool_call_id}: {_text(m.content)}]"
        if last_user is None:
            raise ValueError("RunAgentInput has no user message")
        if trailing and turns and turns[-1][self.input_field] == last_user:
            turns.pop()  # that exchange is carried by the question instead
        question = "\n\n".join([last_user, *trailing])
        kwargs = {self.input_field: question}
        if self.history_field:
            if react_v2:
                turns = [_react_v2_event(t, self.input_field, self.output_field, i) for i, t in enumerate(turns)]
            kwargs[self.history_field] = dspy.History(messages=turns)
        return kwargs


def _react_v2_event(turn: dict[str, str], input_field: str, output_field: str, index: int) -> dict[str, Any]:
    """One past turn in the shape dspy.ReActV2 records: inputs, a submit call with the outputs, and the outputs."""
    from dspy.adapters.types.tool import ToolCallResults, ToolCalls

    outputs = {output_field: turn[output_field]}
    calls = ToolCalls(tool_calls=[ToolCalls.ToolCall(id=f"history-{index}", name="submit", args=outputs)])
    results = ToolCallResults.from_tool_calls_and_values(calls, [outputs], [False])
    return {input_field: turn[input_field], "tool_calls": calls.model_copy(update={"tool_call_results": results}), **outputs}


def _frontend_tools(tools: list[Tool]) -> list[dspy.Tool]:
    out = []
    for t in tools:
        props = (t.parameters or {}).get("properties", {}) or {}

        def stub(_name=t.name, **kwargs):
            raise FrontendToolCall(_name)

        out.append(dspy.Tool(func=stub, name=t.name, desc=t.description or "", args=props))
    return out


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):  # multimodal content parts
        return "".join(getattr(p, "text", "") or "" for p in value)
    try:
        return json.dumps(value, default=str)
    except TypeError:
        return str(value)
