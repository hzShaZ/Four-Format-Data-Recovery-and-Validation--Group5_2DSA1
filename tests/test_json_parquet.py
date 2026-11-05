import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src.readers import CANONICAL, FileLevelError, flatten_record, read_source

try:
    import pyarrow  # noqa: F401
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
