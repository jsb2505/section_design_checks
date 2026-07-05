#!/usr/bin/env python
"""
Test runner script for the materials library.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py -v           # Verbose output
    python run_tests.py -k concrete  # Run only concrete tests
    python run_tests.py --cov        # With coverage report
"""

import sys
import pytest


def main():
    """Run pytest with default configuration."""
    args = sys.argv[1:]  # Pass through any command line arguments

    # Default arguments if none provided
    if not args:
        args = [
            "-v",
            "--tb=short",
            "--cov=materials",
            "--cov-report=html",
            "--cov-report=term-missing",
        ]

    # Run pytest
    exit_code = pytest.main(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
