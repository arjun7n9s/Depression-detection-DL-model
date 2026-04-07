"""WSGI entrypoint for production serving."""

from __future__ import annotations

from src.inference.server import InferenceServerConfig, create_app

app = create_app(InferenceServerConfig())
