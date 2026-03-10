#!/usr/bin/env python3
# Copyright 2025 Miromind.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MiroFlow Agent - Unified CLI Entry Point

Usage:
    # Run a single folder task
    uv run python main.py run --folder data/datasets_en/005 --query "..." --offline

    # Run batch folder tasks
    uv run python main.py batch --data-dir data/datasets_en --context-size 32k --llm-config gpt-4 --run-batch 20260108 --offline
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        print("\nSubcommands:")
        print("  run    Run a single folder task")
        print("  batch  Run batch folder tasks from a JSONL file")
        print("\nUse 'main.py <subcommand> --help' for more details.")
        sys.exit(0)

    subcommand = sys.argv[1]
    # Remove the subcommand from argv so downstream parsers work correctly
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if subcommand == "run":
        from src.runners.folder_task import main as run_main
        run_main()
    elif subcommand == "batch":
        from src.runners.batch_tasks import main as batch_main
        batch_main()
    else:
        print(f"Unknown subcommand: {subcommand}")
        print("Available subcommands: run, batch")
        sys.exit(1)


if __name__ == "__main__":
    main()
