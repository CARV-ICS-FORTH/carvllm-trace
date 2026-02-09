#!/usr/bin/env python3
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
#!/usr/bin/env python3
"""
CLI script that accepts a file path as argument.
"""
import argparse
import os
import sys

from carvllm_trace.utils import read_events_csv, find_type_overlaps


def validate_file_path(filepath):
    """Validate that the file exists."""
    if not os.path.isfile(filepath):
        raise argparse.ArgumentTypeError(f"Error: '{filepath}' is not a valid file")
    return filepath


def process_data(file_path):
    df = read_events_csv(
        file_path,
        start_col="time_range_start_us",
        end_col="time_range_end_us",
        method_col="event_operation",
        delimiter="|"
    )
    comm_overlaps = find_type_overlaps(
        df,
        type_a="nccl",
        type_b="no_nccl"
    )




def main():
    parser = argparse.ArgumentParser(
        description="CLI tool that processes a file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "file_path",
        type=validate_file_path,
        help="Path to the input csv file"
    )

    args = parser.parse_args()

    print(f"Processing file: {args.file_path}")
    print(f"File exists: True")
    print(f"File size: {os.path.getsize(args.file_path)} bytes")
    process_data(args.file_path)


if __name__ == "__main__":
    main()
