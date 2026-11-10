# ag-ui-dspy

[AG-UI](https://docs.ag-ui.com) integration for [DSPy](https://dspy.ai). It serves a `dspy.Module` as an AG-UI agent: the module receives the messages, tools and state an AG-UI client sends, and its run is streamed back as the protocol's events, so CopilotKit or any other AG-UI client can be its front end.

A community integration, maintained here and not by the AG-UI or DSPy teams. It uses the same package layout and names as the integrations in the AG-UI repo: `ag_ui_<framework>`, a `<Framework>Agent`, an `add_<framework>_fastapi_endpoint` helper.

Backend tool calls, streamed answer text, conversation history and frontend (client-executed) tools all map onto the protocol with no changes to DSPy. The bridge uses two hooks DSPy already has: `dspy.streamify` and `BaseCallback`.


## Install

```bash
pip install ag-ui-dspy        # or: uv add ag-ui-dspy
```

## Run the demo

```bash
git clone https://github.com/wrabit/ag-ui-dspy && cd ag-ui-dspy
uv sync --all-groups
cp .env.example .env          # pick a model: any LiteLLM string, local or hosted
set -a; source .env; set +a
uv run uvicorn examples.server:app --port 8040
```

Open http://localhost:8040. The left pane is a chat, the right pane shows every AG-UI event as it arrives. Try:

- "weather in Kraków?" (backend tool)
- "what is 12 * (3 + 4)?" (backend tool)
- "help me pick a city for a weekend" (the agent calls the frontend tool `pick_option`; the page renders buttons; your click goes back as a `ToolMessage` in the next run)

To check the server against the official TypeScript client rather than the demo page:

```bash
cd examples/node-client && npm install && npm start -- "weather in Tokyo?"
```

From CopilotKit or any other AG-UI client, point an `HttpAgent` at the endpoint URL. No client-side package from this repo is needed.

## Use it on your own module

```python
import dspy
from fastapi import FastAPI
from ag_ui_dspy import DSPyAgent, add_dspy_fastapi_endpoint

def make_agent(frontend_tools: list[dspy.Tool]) -> dspy.Module:
    # frontend_tools are the client's tools from RunAgentInput.tools, already
    # wrapped as dspy.Tool stubs; include them if you want the LM to use them.
    return dspy.ReActV2("question -> answer", tools=[my_tool, *frontend_tools])

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
app = FastAPI()
add_dspy_fastapi_endpoint(app, DSPyAgent(make_agent), path="/agent")
```

`add_dspy_fastapi_endpoint` is a convenience, named like `add_langgraph_fastapi_endpoint` and `add_adk_fastapi_endpoint` in the AG-UI repo's own integrations. This is the route it registers; write it yourself when you need auth, middleware or another framework:

```python
from ag_ui.core import RunAgentInput
from ag_ui.encoder import EventEncoder
from fastapi import Request
from fastapi.responses import StreamingResponse

agent = DSPyAgent(make_agent)

@app.post("/agent")
async def agent_endpoint(run_input: RunAgentInput, request: Request):
    encoder = EventEncoder(accept=request.headers.get("accept"))

    async def events():
        async for event in agent.run(run_input):   # AsyncIterator[ag_ui.core.BaseEvent]
            yield encoder.encode(event)             # "data: {...}\n\n"

    return StreamingResponse(events(), media_type=encoder.get_content_type())
```

`DSPyAgent.run(run_input)` takes a `RunAgentInput` and returns an async iterator of `ag_ui.core` events. Anything that can stream server-sent events can serve it.

`DSPyAgent(program_factory, input_field="question", output_field="answer", history_field="history")`:

- `input_field` receives the latest user message.
- `output_field` is the assistant reply: streamed token by token when a predictor generates it as text, sent whole when the agent loop returns it.
- `history_field`, if set, receives earlier turns as a `dspy.History`. Set it to `None` for a stateless module.

The factory is called once per run so each run sees that request's frontend tools.

The agent loop is `dspy.ReActV2`. The adapter passes the conversation as its `history` argument in the format the loop records its own turns in, so the model sees earlier replies, keeps its internal `submit` tool out of the tool events, and emits the answer when the loop finishes. DSPy is pinned to the 3.4 series because 3.5 renames that loop to `dspy.ReAct`; the pin moves with the adapter.

Any other module works too. A plain `dspy.Predict` or `ChainOfThought` streams its answer and never emits tool events, which makes it the quickest way to see the plumbing before adding tools:

```python
DSPyAgent(lambda frontend_tools: dspy.Predict("question, history: dspy.History -> answer"))
```

## How the mapping works

| DSPy | AG-UI |
|---|---|
| `BaseCallback.on_tool_start` | `TOOL_CALL_START`, `TOOL_CALL_ARGS`, `TOOL_CALL_END` |
| `BaseCallback.on_tool_end` | `TOOL_CALL_RESULT` |
| `StreamListener` chunk on the output field | `TEXT_MESSAGE_START` once, then `TEXT_MESSAGE_CONTENT` (listener attached only when exactly one predictor generates the field) |
| final `dspy.Prediction` | `TEXT_MESSAGE_END` (whole answer first if nothing streamed: the agent loop's `submit`, a cache hit, a field shared by several predictors or post-processed in Python) |
| `StatusMessage` | `STEP_STARTED` |
| frontend tool chosen by the LM | `TOOL_CALL_*` then `RUN_FINISHED` with no result; the client executes it and starts a new run |
| exception | `RUN_ERROR` |

`dspy.streamify` runs the program with a send stream in its settings and passes through any object it does not recognise. `AgUiCallback` drops small `ToolStart`/`ToolEnd` dataclasses onto that stream, so the adapter gets structured tool events with DSPy's own `call_id`, not parsed status strings.

Frontend tools become `dspy.Tool` stubs that raise `FrontendToolCall`. It subclasses `BaseException` because the agent loop catches `Exception` and would record it as an "Execution error" observation instead of stopping.

History: the AG-UI message list is folded into `dspy.History` turns of `{input_field, output_field}`. Assistant tool calls and tool results are appended to the previous turn's answer as text, so a run after a frontend tool round-trip starts a fresh trajectory that can see what happened.

## Not covered yet

- Shared state (`STATE_SNAPSHOT` / `STATE_DELTA`). DSPy doesn't have a state concept, so `RunAgentInput.state` is ignored and no state events are emitted. Supporting it would mean passing the state into a `state` input field and emitting a snapshot when the module returns one.
- When a frontend tool interrupts a turn, the tool calls made before it are passed to the next run as text in the history. If you are using ReActV2, it records its own tool calls in a structured form, and the adapter doesn't yet rebuild them in that form.

## Tests

```bash
uv run pytest
```

The tests run against `dspy.utils.dummies.DummyLM`, so no network or key is needed.

## Releasing

Bump `version` in `pyproject.toml`, commit, then `git tag vX.Y.Z && git push origin main vX.Y.Z`. The publish workflow builds, tests and uploads to PyPI through trusted publishing.

## License

MIT
