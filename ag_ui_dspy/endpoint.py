"""FastAPI mounting, the same shape every Python AG-UI integration uses.

This is a convenience. The route it registers is three lines, shown in the
README, so write it yourself when you need middleware, auth or a different
framework.
"""

from __future__ import annotations

from typing import Any

from ag_ui.core import RunAgentInput
from ag_ui.encoder import EventEncoder
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import StreamingResponse

from .adapter import DSPyAgent


def add_dspy_fastapi_endpoint(
    app: FastAPI | APIRouter, agent: DSPyAgent, path: str = "/", **kwargs: Any
) -> None:
    """Register ``POST path`` on ``app`` serving ``agent`` as an AG-UI event stream.

    ``kwargs`` go to ``app.post`` (``tags``, ``dependencies``, ``name`` ...).
    """

    @app.post(path, **kwargs)
    async def dspy_agent_endpoint(run_input: RunAgentInput, request: Request):
        encoder = EventEncoder(accept=request.headers.get("accept"))

        async def events():
            async for event in agent.run(run_input):
                yield encoder.encode(event)

        return StreamingResponse(events(), media_type=encoder.get_content_type())
