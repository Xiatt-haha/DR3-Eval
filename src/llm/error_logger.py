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

"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# Path(__file__) = apps/miroflow-agent/src/llm/error_logger.py
# .parent = apps/miroflow-agent/src/llm
# .parent.parent = apps/miroflow-agent/src
# .parent.parent.parent = apps/miroflow-agent
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

API_ERROR_LOG_FILE = LOG_DIR / "api_errors.log"

api_error_logger = logging.getLogger("api_error_logger")
api_error_logger.setLevel(logging.INFO)

if not api_error_logger.handlers:
    file_handler = logging.FileHandler(API_ERROR_LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    
    api_error_logger.addHandler(file_handler)
    
    api_error_logger.propagate = False

def log_api_error(
    error_type: str,
    model_name: str,
    task_id: str,
    error_message: str,
    trace_id: Optional[str] = None,
    original_error: Optional[Exception] = None,
    extra_info: Optional[dict] = None
) -> None:
    """
    
    Args:
    """
    parts = [
        f"ErrorType: {error_type}",
        f"Model: {model_name}",
    ]
    
    if extra_info:
        if extra_info.get("context_size"):
            parts.append(f"ContextSize: {extra_info['context_size']}")
        if extra_info.get("dataset"):
            parts.append(f"Dataset: {extra_info['dataset']}")
    
    parts.append(f"CaseID: {task_id}")
    
    if trace_id:
        parts.append(f"TraceID: {trace_id}")
    
    parts.append(f"Message: {error_message}")
    
    if original_error:
        error_str = str(original_error)[:500]
        parts.append(f"OriginalError: {type(original_error).__name__}: {error_str}")
    
    if extra_info:
        for key, value in extra_info.items():
            if key not in ("context_size", "dataset"):
                parts.append(f"{key}: {value}")
    
    log_message = " | ".join(parts)
    api_error_logger.error(log_message)

def log_api_skip(
    model_name: str,
    task_id: str,
    reason: str,
    trace_id: Optional[str] = None,
    extra_info: Optional[dict] = None
) -> None:
    """
    
    Args:
    """
    parts = [
        "TASK_SKIPPED",
        f"Model: {model_name}",
    ]
    
    if extra_info:
        if extra_info.get("context_size"):
            parts.append(f"ContextSize: {extra_info['context_size']}")
        if extra_info.get("dataset"):
            parts.append(f"Dataset: {extra_info['dataset']}")
    
    parts.append(f"CaseID: {task_id}")
    parts.append(f"Reason: {reason}")
    
    if trace_id:
        parts.append(f"TraceID: {trace_id}")
    
    log_message = " | ".join(parts)
    api_error_logger.warning(log_message)

def get_log_file_path() -> str:
    return str(API_ERROR_LOG_FILE)

api_error_logger.info(f"API Error Logger initialized. Log file: {API_ERROR_LOG_FILE}")
