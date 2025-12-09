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
"""carvllm-trace testing package."""

import pathlib

import toml

import carvllm_trace


REPO_PATH = pathlib.Path(__file__).parents[1]


def test_version() -> None:
    """Check that all the version tags are in sync."""
    toml_path = REPO_PATH / "pyproject.toml"
    expected = toml.load(toml_path)["project"]["version"]

    actual = carvllm_trace.__version__
    assert actual == expected
