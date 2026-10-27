"""The FastAPI route end to end, with DummyLM behind it."""

import json

import dspy
from dspy.utils.dummies import DummyLM
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ag_ui_dspy import DSPyAgent, add_dspy_fastapi_endpoint


def test_post_returns_sse_event_stream():
    dspy.configure(lm=DummyLM([{"answer": "forty-two"}]), cache=False)
    app = FastAPI()
    add_dspy_fastapi_endpoint(app, DSPyAgent(lambda tools: dspy.Predict("question -> answer"), history_field=None), path="/agent")

    body = {"threadId": "t", "runId": "r", "state": {}, "messages": [{"id": "u1", "role": "user", "content": "meaning of life?"}],
            "tools": [], "context": [], "forwardedProps": {}}
    with TestClient(app) as client:
        res = client.post("/agent", json=body, headers={"accept": "text/event-stream"})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(line[5:]) for line in res.text.splitlines() if line.startswith("data:")]
    assert [e["type"] for e in events] == ["RUN_STARTED", "TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT", "TEXT_MESSAGE_END", "RUN_FINISHED"]
    assert events[2]["delta"] == "forty-two"
