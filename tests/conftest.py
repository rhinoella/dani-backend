"""
Root pytest configuration for DANI-Engine tests.

This file registers custom command line options and markers that can be used
across all test directories.
"""

import pytest


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests (requires external services like Qdrant)",
    )
    parser.addoption(
        "--performance",
        action="store_true",
        default=False,
        help="Run performance tests",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require external services)"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow"
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration/performance tests unless flags are provided."""
    run_integration = config.getoption("--integration")
    run_performance = config.getoption("--performance")
    
    skip_integration = pytest.mark.skip(
        reason="Need --integration option to run integration tests"
    )
    skip_performance = pytest.mark.skip(
        reason="Need --performance option to run performance tests"
    )
    
    for item in items:
        if "integration" in item.keywords and not run_integration:
            item.add_marker(skip_integration)
        if "performance" in item.keywords and not run_performance:
            item.add_marker(skip_performance)
