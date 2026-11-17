"""OpenTelemetry spans per agent node, carrying tokens and rupees as attributes.

The console exporter is off by default (``holdout agent --trace`` turns it on); tests can
attach an in-memory exporter. Spans never carry prompt text, data or the API key.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)

_PROVIDER: TracerProvider | None = None


def configure_tracing(
    *, console: bool = False, exporter: SpanExporter | None = None
) -> TracerProvider:
    """Install the tracer provider once; add the console and/or a given exporter."""
    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = TracerProvider()
        trace.set_tracer_provider(_PROVIDER)
    if console:
        _PROVIDER.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    if exporter is not None:
        _PROVIDER.add_span_processor(SimpleSpanProcessor(exporter))
    return _PROVIDER


def tracer() -> trace.Tracer:
    return trace.get_tracer("holdout.agent")
