#!/bin/bash
# Run batch folder tasks for Claude 3.5 Sonnet
# All results will be saved to: result/<run_batch>/<dataset>/<context_size>/<model>/
export ENABLE_REPORT_VALIDATION="0"
RUN_BATCH="20260108"
DATA_DIR="data/datasets_en"

# claude35_sonnet
uv run python main.py batch --data-dir $DATA_DIR --context-size 32k --llm-config claude35_sonnet --run-batch $RUN_BATCH --offline
uv run python main.py batch --data-dir $DATA_DIR --context-size 64k --llm-config claude35_sonnet --run-batch $RUN_BATCH --offline
# uv run python main.py batch --data-dir $DATA_DIR --context-size 128k --llm-config claude35_sonnet --run-batch $RUN_BATCH --offline
# uv run python main.py batch --data-dir $DATA_DIR --context-size 256k --llm-config claude35_sonnet --run-batch $RUN_BATCH --offline
# uv run python main.py batch --data-dir $DATA_DIR --context-size 512k --llm-config claude35_sonnet --run-batch $RUN_BATCH --offline
