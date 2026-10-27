"""FastAPI server: the AG-UI endpoint at /agent and the demo chat page at /.

    uv run uvicorn examples.server:app --port 8040
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from ag_ui_dspy import DSPyAgent, add_dspy_fastapi_endpoint

from .agent import configure_lm, make_agent

_load_dotenv = None
try:  # optional: python-dotenv is not a dependency
    from dotenv import load_dotenv as _load_dotenv
except ImportError:
    pass
if _load_dotenv:
    _load_dotenv()

configure_lm()

app = FastAPI(title="ag-ui-dspy demo")
agent = DSPyAgent(make_agent, input_field="question", output_field="answer", history_field="history")
add_dspy_fastapi_endpoint(app, agent, path="/agent")

STATIC = Path(__file__).parent / "static"


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("examples.server:app", host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8040")))
