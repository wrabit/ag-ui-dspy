"""FastAPI mounting, the same shape every Python AG-UI integration uses."""

from __future__ import annotations

from ag_ui.core import RunAgentInput
from ag_ui.encoder import EventEncoder
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from .adapter import DSPyAgent


def add_dspy_fastapi_endpoint(app: FastAPI, agent: DSPyAgent, path: str = "/") -> None:
    @app.post(path)
    async def agui_endpoint(run_input: RunAgentInput, request: Request):
        encoder = EventEncoder(accept=request.headers.get("accept"))

        async def events():
            async for event in agent.run(run_input):
                yield encoder.encode(event)

        return StreamingResponse(events(), media_type=encoder.get_content_type())
