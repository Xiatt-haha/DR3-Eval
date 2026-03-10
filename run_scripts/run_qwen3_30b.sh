#!/bin/bash
# Run batch folder tasks for Qwen3 30B
# All results will be saved to: result/<run_batch>/<dataset>/<context_size>/<model>/
export ENABLE_REPORT_VALIDATION="0"
RUN_BATCH="20260108"
DATA_DIR="data/datasets_en"

# qwen3_30b
uv run python main.py batch --data-dir $DATA_DIR --context-size 32k --llm-config qwen3_30b --run-batch $RUN_BATCH --offline
uv run python main.py batch --data-dir $DATA_DIR --context-size 64k --llm-config qwen3_30b --run-batch $RUN_BATCH --offline
uv run python main.py batch --data-dir $DATA_DIR --context-size 128k --llm-config qwen3_30b --run-batch $RUN_BATCH --offline
uv run python main.py batch --data-dir $DATA_DIR --context-size 256k --llm-config qwen3_30b --run-batch $RUN_BATCH --offline
uv run python main.py batch --data-dir $DATA_DIR --context-size 512k --llm-config qwen3_30b --run-batch $RUN_BATCH --offline
