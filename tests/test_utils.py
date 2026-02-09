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

from carvllm_trace.utils import find_type_overlaps, measure_percentage_overlapping


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




def test_single_full_overlap():
    df = pd.DataFrame(
        {
            "a_start": [0],
            "a_end": [10],
            "b_start": [0],
            "b_end": [10],
        }
    )

    result = measure_percentage_overlapping(df)

    expected = pd.DataFrame(
        {
            "a_start": [0],
            "a_end": [10],
            "overlapping": [1.0],
        }
    )

    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


def test_single_partial_overlap():
    df = pd.DataFrame(
        {
            "a_start": [0],
            "a_end": [10],
            "b_start": [5],
            "b_end": [10],
        }
    )

    result = measure_percentage_overlapping(df)

    expected = pd.DataFrame(
        {
            "a_start": [0],
            "a_end": [10],
            "overlapping": [0.5],
        }
    )

    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


def test_single_a_interval_with_no_overlap():
    df = pd.DataFrame(
        {
            "a_start": [],
            "a_end": [],
            "b_start": [],
            "b_end": [],
        }
    )

    result = measure_percentage_overlapping(df)

    expected = pd.DataFrame(columns=["a_start", "a_end", "overlapping"])

    pd.testing.assert_frame_equal(result, expected)


def test_multiple_a_intervals_some_overlap_some_zero():
    # A intervals: [0,10], [20,30]
    # Only first has overlap of 5
    df = pd.DataFrame(
        {
            "a_start": [0],
            "a_end": [10],
            "b_start": [5],
            "b_end": [10],
        }
    )

    # We expect both A intervals to appear; second has zero overlap
    # So we simulate presence of second A interval with no B by adding a zero-length overlap row
    df_full = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "a_start": [20],
                    "a_end": [30],
                    "b_start": [None],
                    "b_end": [None],
                }
            ),
        ],
        ignore_index=True,
    )

    result = measure_percentage_overlapping(df_full)

    expected = pd.DataFrame(
        {
            "a_start": [0, 20],
            "a_end": [10, 30],
            "overlapping": [0.5, 0.0],
        }
    )

    pd.testing.assert_frame_equal(result.sort_values(["a_start"]).reset_index(drop=True),
                                  expected.sort_values(["a_start"]).reset_index(drop=True))


def test_multiple_entries_for_same_a_interval_are_merged_without_double_counting():
    # A interval [0,10] appears twice with two disjoint overlaps: [0,4] and [6,10]
    # Total overlap = 4 + 4 = 8 -> fraction = 0.8
    df = pd.DataFrame(
        {
            "a_start": [0, 0],
            "a_end": [10, 10],
            "b_start": [0, 6],
            "b_end": [4, 10],
        }
    )

    result = measure_percentage_overlapping(df)

    expected = pd.DataFrame(
        {
            "a_start": [0],
            "a_end": [10],
            "overlapping": [0.8],
        }
    )

    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


def test_overlapping_b_intervals_on_same_a_do_not_double_count():
    # A interval [0,10]
    # B overlaps: [2,8] and [4,9] -> union is [2,9] length 7 -> fraction = 0.7
    df = pd.DataFrame(
        {
            "a_start": [0, 0],
            "a_end": [10, 10],
            "b_start": [2, 4],
            "b_end": [8, 9],
        }
    )

    result = measure_percentage_overlapping(df)

    expected = pd.DataFrame(
        {
            "a_start": [0],
            "a_end": [10],
            "overlapping": [0.7],
        }
    )

    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


def test_touching_intervals_do_not_count_as_overlap():
    # A [0,10], B [10,20] -> touching only, no overlap
    df = pd.DataFrame(
        {
            "a_start": [0],
            "a_end": [10],
            "b_start": [10],
            "b_end": [20],
        }
    )

    result = measure_percentage_overlapping(df)

    expected = pd.DataFrame(
        {
            "a_start": [0],
            "a_end": [10],
            "overlapping": [0.0],
        }
    )

    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


def test_multiple_a_intervals_with_multiple_overlaps():
    # A intervals: [0,10], [20,30]
    # Overlaps:
    #   [0,10] with [2,5] -> 3
    #   [0,10] with [7,9] -> 2  => total 5 -> 0.5
    #   [20,30] with [25,35] -> 5 -> 0.5
    df = pd.DataFrame(
        {
            "a_start": [0, 0, 20],
            "a_end": [10, 10, 30],
            "b_start": [2, 7, 25],
            "b_end": [5, 9, 35],
        }
    )

    result = measure_percentage_overlapping(df)

    expected = pd.DataFrame(
        {
            "a_start": [0, 20],
            "a_end": [10, 30],
            "overlapping": [0.5, 0.5],
        }
    )

    pd.testing.assert_frame_equal(
        result.sort_values(["a_start"]).reset_index(drop=True),
        expected.sort_values(["a_start"]).reset_index(drop=True),
    )
