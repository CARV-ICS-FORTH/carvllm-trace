#   Copyright 2025 - 2025 Christos Kozanitis, FORTH, Greece
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

@pytest.fixture
def file_factory(tmp_path):
    """
Returns a function to create files with arbitrary content.
    """
    def _create_csv(content: str, filename: str = "trace.txt") -> Path:
        file_path = tmp_path / filename
        file_path.write_text(content)
        return file_path
    return _create_csv

def test_load_file(file_factory):
    content = '\n'.join([
        '[prof-debug] #1 | name=Memcpy HtoD (Pinned -> Device) | device_us=1393.915 | cpu_us=0.0 | time_range_start_us=942.737 | time_range_end_us=2336.652',
        '[prof-debug] #2 | name=Memcpy HtoD (Pinned -> Device) | device_us=0.6080000000001746 | cpu_us=0.0 | time_range_start_us=3317.352 | time_range_end_us=3317.96',
        '[prof-debug] #3 | name=ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | device_us=6177.607000000018 | cpu_us=0.0 | time_range_start_us=363777.119 | time_range_end_us=369954.726',
        '[prof-debug] #4 | name=Memset (Device) | device_us=0.8319999999948777 | cpu_us=0.0 | time_range_start_us=300908.284 | time_range_end_us=300909.116',
        '[prof-debug] #5 | name=ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | device_us=13942.407999999938 | cpu_us=0.0 | time_range_start_us=829450.831 | time_range_end_us=843393.239',
        '[prof-debug] #6 | name=Memcpy HtoD (Pageable -> Device) | device_us=0.31999999994877726 | cpu_us=0.0 | time_range_start_us=301059.579 | time_range_end_us=301059.899',
        '[prof-debug] #7 | name=Memcpy HtoD (Pageable -> Device) | device_us=0.31999999994877726 | cpu_us=0.0 | time_range_start_us=301059.579 | time_range_end_us=301059.899',
        ]
    )
    file_path = file_factory(content)
    timeline = utils.parse_trace(file_path)
    expected_answers = [
        [942.737, 2336.652, utils.event_type.COMPUTE_OR_MEM],
        [3317.352, 3317.96, utils.event_type.COMPUTE_OR_MEM],
        [363777.119, 369954.726, utils.event_type.NCCL],
        [300908.284, 300909.116, utils.event_type.COMPUTE_OR_MEM],
        [829450.831, 843393.239, utils.event_type.NCCL],
        [301059.579, 301059.899, utils.event_type.COMPUTE_OR_MEM],
        [301059.579, 301059.899, utils.event_type.COMPUTE_OR_MEM]
    ]
    for e, t in zip_longest(expected_answers, timeline):
        assert e==t

@pytest.mark.parameterize(
    "interval1, interval2, expected",
    [
        ([10,12], [28,30], [[10, 12], [28, 30]]),
        ([10,15], [13, 14], [[10, 15]]),
        ([10, 15], [14, 17], [[10, 17]]),
        ([10, 15], [9, 11], [[9, 15]]),
    ]
)
def test_merge_intervals_parameterized(interval1, interval2, expected):
    result = utils.merge_intervals(interval1, interval2)
    assert result == expected

@pytest.mark.parameterize(
    "interval1, interval2, expected",
    [
        ([10,12], [28,30], False),
        ([10,15], [13, 14], True),
        ([10, 15], [14, 17], True),
        ([10, 15], [9, 11], True),
    ]
)
def test_intervals_overlap_parameterized(interval1, interval2, expected):
    result = utils.intervals_overlap(interval1, interval2)
    assert result == expected

@pytest.mark.parameterize(
    "interval1, interval2, expected",
    [
        ([10,12], [28,30], 0),
        ([10,15], [13, 14], 2), #positions 13,14
        ([10, 15], [14, 17], 2), #positions 14,15
        ([10, 15], [9, 11], 7), #positions: 15-9+1
    ]
)
def test_measure_overlapping_parameterized(interval1, interval2, expected):
    result = utils.measure_overlapping(interval1, interval2)
    assert result == expected

def test_overlap_pipelines():
    index_timeline = [
        [10, 15],
        [15, 20],
        [20, 25],
        [25, 30]
    ]
    target_timeline = [
        [12, 14],
        [16, 17],
        [17, 18],
        [20, 24],
        [23, 28]
    ]
    expected = [
        [10, 15, 3], #12, 13, 14
        [15, 20, 3], #16, 17, 19
        [20, 25, 6], #25-20+1
        [25, 30, 4], #25-28
    ]
    result = utils.overlap_pipelines(index_timeline, target_timeline)
    for e, r in zip_longest(expected, result):
        assert e == r
