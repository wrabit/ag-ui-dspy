"""Event-stream contract, checked against dspy's DummyLM (no network)."""

import asyncio
import json

import dspy
import pytest
from ag_ui.core import AssistantMessage, FunctionCall, RunAgentInput, Tool, ToolCall, ToolMessage, UserMessage
from dspy.utils.dummies import DummyLM

from ag_ui_dspy import DSPyAgent


def get_weather(city: str) -> str:
    """Weather for a city."""
    return f"Sunny in {city}"


def make(frontend_tools):
    return dspy.ReAct("question, history: dspy.History -> answer", tools=[get_weather, *frontend_tools])


PICK = Tool(name="pick_city", description="ask the user to pick a city",
            parameters={"type": "object", "properties": {"options": {"type": "array"}}})


def run(answers, messages, tools=()):
    dspy.configure(lm=DummyLM(answers), cache=False)
    agent = DSPyAgent(make)
    ri = RunAgentInput(thread_id="t1", run_id="r1", state={}, messages=messages, tools=list(tools), context=[], forwarded_props={})

    async def collect():
        return [ev async for ev in agent.run(ri)]

    return asyncio.run(collect())


def types(evs):
    return [e.type.value for e in evs]


def test_backend_tool_then_answer_with_history():
    evs = run(
        [
            {"next_thought": "look it up", "next_tool_name": "get_weather", "next_tool_args": {"city": "Kraków"}},
            {"next_thought": "done", "next_tool_name": "finish", "next_tool_args": {}},
            {"reasoning": "r", "answer": "It is sunny in Kraków."},
        ],
        [
            UserMessage(id="m1", role="user", content="hi"),
            AssistantMessage(id="m2", role="assistant", content="hello"),
            UserMessage(id="m3", role="user", content="weather in Kraków?"),
        ],
    )
    assert types(evs) == [
        "RUN_STARTED", "TOOL_CALL_START", "TOOL_CALL_ARGS", "TOOL_CALL_END", "TOOL_CALL_RESULT",
        "TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT", "TEXT_MESSAGE_END", "RUN_FINISHED",
    ]
    assert evs[1].tool_call_name == "get_weather"
    assert json.loads(evs[2].delta) == {"city": "Kraków"}
    assert evs[4].content == "Sunny in Kraków"
    assert evs[6].delta == "It is sunny in Kraków."


def test_frontend_tool_ends_the_run_without_a_result():
    evs = run(
        [{"next_thought": "ask", "next_tool_name": "pick_city", "next_tool_args": {"options": ["a", "b"]}}],
        [UserMessage(id="m1", role="user", content="weather?")],
        tools=[PICK],
    )
    assert types(evs) == ["RUN_STARTED", "TOOL_CALL_START", "TOOL_CALL_ARGS", "TOOL_CALL_END", "RUN_FINISHED"]
    assert evs[1].tool_call_name == "pick_city"
    assert json.loads(evs[2].delta) == {"options": ["a", "b"]}


def test_continuation_after_client_tool_result():
    evs = run(
        [
            {"next_thought": "done", "next_tool_name": "finish", "next_tool_args": {}},
            {"reasoning": "r", "answer": "Sunny in a."},
        ],
        [
            UserMessage(id="m1", role="user", content="weather?"),
            AssistantMessage(id="m2", role="assistant", content=None, tool_calls=[
                ToolCall(id="c1", type="function", function=FunctionCall(name="pick_city", arguments='{"options":["a","b"]}'))]),
            ToolMessage(id="m3", role="tool", tool_call_id="c1", content="a"),
        ],
        tools=[PICK],
    )
    assert types(evs)[-2:] == ["TEXT_MESSAGE_END", "RUN_FINISHED"]


def test_history_mapping_folds_tool_exchange_into_previous_turn():
    agent = DSPyAgent(make)
    kwargs = agent._inputs([
        UserMessage(id="m1", role="user", content="weather?"),
        AssistantMessage(id="m2", role="assistant", content=None, tool_calls=[
            ToolCall(id="c1", type="function", function=FunctionCall(name="pick_city", arguments='{"options":["a","b"]}'))]),
        ToolMessage(id="m3", role="tool", tool_call_id="c1", content="a"),
    ])
    assert kwargs["question"].startswith("weather?")
    assert "pick_city" in kwargs["question"] and "the result was: a" in kwargs["question"]
    assert kwargs["history"].messages == []


def test_history_mapping_keeps_completed_turns():
    agent = DSPyAgent(make)
    kwargs = agent._inputs([
        UserMessage(id="m1", role="user", content="hi"),
        AssistantMessage(id="m2", role="assistant", content="hello"),
        UserMessage(id="m3", role="user", content="weather?"),
    ])
    assert kwargs["question"] == "weather?"
    assert kwargs["history"].messages == [{"question": "hi", "answer": "hello"}]


def test_no_user_message_is_an_error():
    with pytest.raises(ValueError):
        DSPyAgent(make)._inputs([])
