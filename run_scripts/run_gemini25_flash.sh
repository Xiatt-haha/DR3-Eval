#!/bin/bash
# Run batch folder tasks for Gemini 2.5 Flash 06-17
# All results will be saved to: result/<run_batch>/<dataset>/<context_size>/<model>/
export ENABLE_REPORT_VALIDATION="0"
RUN_BATCH="20260108"
DATA_DIR="data/datasets_en"
LLM_CONFIG="gemini-2.5-flash-06-17"

uv run python main.py batch --data-dir $DATA_DIR --context-size 32k --llm-config $LLM_CONFIG --run-batch $RUN_BATCH --offline
uv run python main.py batch --data-dir $DATA_DIR --context-size 64k --llm-config $LLM_CONFIG --run-batch $RUN_BATCH --offline
uv run python main.py batch --data-dir $DATA_DIR --context-size 128k --llm-config $LLM_CONFIG --run-batch $RUN_BATCH --offline
uv run python main.py batch --data-dir $DATA_DIR --context-size 256k --llm-config $LLM_CONFIG --run-batch $RUN_BATCH --offline
uv run python main.py batch --data-dir $DATA_DIR --context-size 512k --llm-config $LLM_CONFIG --run-batch $RUN_BATCH --offline
