"""
conftest.py - Pytest root configuration.
Ensures all tests run from the project root regardless of where pytest is invoked.
"""

import os
import pytest


@pytest.fixture(autouse=True)
def set_working_directory():
    root = os.path.abspath(os.path.dirname(__file__))
    os.chdir(root)
