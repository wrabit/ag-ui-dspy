"""AG-UI for DSPy: expose any dspy.Module as an AG-UI agent."""

from .adapter import DSPyAgent
from .callback import FrontendToolCall
from .endpoint import add_dspy_fastapi_endpoint

__all__ = ["DSPyAgent", "FrontendToolCall", "add_dspy_fastapi_endpoint"]
