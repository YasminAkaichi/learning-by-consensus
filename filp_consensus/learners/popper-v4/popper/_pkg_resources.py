"""Minimal replacement for the two `pkg_resources` functions Popper uses.

`pkg_resources` (setuptools) is no longer installed by default with recent
Python / setuptools versions. Resources are plain files next to the modules.
"""

import sys
from pathlib import Path


def _path(module_name, relative):
    module = sys.modules[module_name]
    return Path(module.__file__).resolve().parent / relative


def resource_filename(module_name, relative):
    return str(_path(module_name, relative))


def resource_string(module_name, relative):
    return _path(module_name, relative).read_bytes()
