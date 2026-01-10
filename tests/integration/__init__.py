"""
Integration tests package.

These tests require external services (Qdrant, Ollama) and are skipped by default.
Run with: pytest tests/integration/ -v --integration
"""

import pytest


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires external services)"
    )
