#   Copyright 2025 - 2026 Christos Kozanitis, FORTH, Greece
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
from carvllm_trace import utils

# tests/conftest.py or tests/test_parser.py
import pytest
from pathlib import Path
from itertools import zip_longest

import pandas as pd

def test_same_type_intervals_are_merged():
    df = pd.DataFrame(
        {
            "start": [1, 4],
            "end": [5, 10],
            "type": ["A", "A"],
        }
    )
    result = find_type_overlaps(df, "A", "B")
    # No B intervals → no overlaps,
    # but merge must not explode
    assert result.empty

def test_no_overlap_between_types():
    df = pd.DataFrame(
        {
            "start": [1, 10],
            "end": [5, 15],
            "type": ["A", "B"],
        }
    )

    result = find_type_overlaps(df, "A", "B")

    assert result.empty


def test_overlap_after_merge():
    df = pd.DataFrame(
        {
            "start": [1, 4, 6],
            "end":   [5, 8, 10],
            "type":  ["A", "A", "B"],
        }
    )

    result = find_type_overlaps(df, "A", "B")

    expected = pd.DataFrame(
        {
            "a_start": [1],
            "a_end": [8],
            "b_start": [6],
            "b_end": [10],
        }
    )
    pd.testing.assert_frame_equal(result, expected)

def test_multiple_overlappings():
    df = pd.DataFrame(
        {
            "start": [1, 10, 4],
            "end":   [5, 12, 22],
            "type":  ["A", "A", "B"],
        }
    )

    result = find_type_overlaps(df, "A", "B")

    expected = pd.DataFrame(
        {
            "a_start": [1, 10],
            "a_end": [5, 12],
            "b_start": [4, 4],
            "b_end": [22, 22],
        }
    )
    pd.testing.assert_frame_equal(result, expected)

def test_touching_intervals_do_not_overlap():
    df = pd.DataFrame(
        {
            "start": [1, 5],
            "end": [5, 10],
            "type": ["A", "B"],
        }
    )

    result = find_type_overlaps(df, "A", "B")

    assert result.empty
