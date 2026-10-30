"""The demo agent: a dspy.ReActV2 with two backend tools.

Frontend tools arrive per request from the AG-UI client (see static/index.html,
which declares ``pick_option``). The factory receives them already wrapped as
dspy.Tool stubs and adds them to the tool list, so the LM can choose to hand a
question back to the user.
"""

from __future__ import annotations

import ast
import operator
import os

import dspy

_WEATHER = {
    "krakow": "12°C, overcast, light wind",
    "kraków": "12°C, overcast, light wind",
    "london": "9°C, drizzle",
    "tenerife": "24°C, sunny",
    "tokyo": "18°C, clear",
}


def get_weather(city: str) -> str:
    """Current weather for a city (demo data, a handful of cities known)."""
    return _WEATHER.get(city.strip().lower(), f"No weather data for {city}; try Kraków, London, Tenerife or Tokyo.")


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression such as '12 * (3 + 4)'."""

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError(f"unsupported expression: {expression}")

    return str(ev(ast.parse(expression, mode="eval")))


class Assistant(dspy.Signature):
    """You are a concise, friendly assistant. Use tools when they help.
    If a question needs a choice from the user and a frontend tool for that exists, call it."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def make_agent(frontend_tools: list[dspy.Tool]) -> dspy.Module:
    # ReActV2 keeps the conversation in its own `history` forward argument (the adapter passes it),
    # so the signature must not declare a history field; it also reserves the `tools` input name.
    return dspy.ReActV2(Assistant, tools=[get_weather, calculate, *frontend_tools], max_iters=6)


def configure_lm() -> dspy.LM:
    """One dspy.LM from the environment: DSPY_LM, DSPY_API_BASE, DSPY_API_KEY."""
    model = os.environ.get("DSPY_LM", "openai/gpt-4o-mini")
    extra = {k: os.environ[e] for k, e in (("api_base", "DSPY_API_BASE"), ("api_key", "DSPY_API_KEY")) if os.environ.get(e)}
    lm = dspy.LM(model, cache=False, max_tokens=2048, **extra)
    dspy.configure(lm=lm)
    return lm
