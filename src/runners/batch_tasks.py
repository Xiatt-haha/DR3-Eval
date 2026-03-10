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
Batch Folder Task Runner

This script processes multiple folders in batch by reading task definitions
from a JSONL file and running each task using run_folder_task.py.

Usage:
    # Run all tasks from query.jsonl
    uv run python run_batch_folder_tasks.py --data-dir data/bench_case1104
    
    # Run specific tasks by number
    uv run python run_batch_folder_tasks.py --data-dir data/bench_case1104 --tasks 001 002 003
    
    # Skip already completed tasks
    uv run python run_batch_folder_tasks.py --data-dir data/bench_case1104 --skip-completed
    
    # Preview tasks without running
    uv run python run_batch_folder_tasks.py --data-dir data/bench_case1104 --preview
    
    # Run with specific context size (32k, 64k, 128k)
    uv run python run_batch_folder_tasks.py --data-dir datasets --context-size 32k --model gpt4.1
    
    # Continue a previous batch run (use the same batch ID)
    uv run python run_batch_folder_tasks.py --data-dir datasets --context-size 32k --model gpt4.1 --run-batch 20251210_150000 --skip-completed

Directory Structure:
    New format: result-<run_batch>-<dataset>-<context_size>-<model>/<task_number>/
    
    Example:
    result-20251222_220000-datasets_batch2-32k-gpt4.1/
    ├── 001/
    │   ├── initial_report.md
    │   ├── final_report.md
    │   └── execution_log.json
    ├── 002/
    │   ├── initial_report.md
    │   ├── final_report.md
    │   └── execution_log.json
    └── ...
    
    This flat structure makes it easy to:
    - Package and share results with others
    - Identify the batch run, dataset, context size, and model at a glance
    - Compare results across different configurations

Output Files:
    For each task run, a unique folder is created containing:
    - initial_report.md: The original report before validation
    - final_report.md: The final report after validation
    - execution_log.json: Full execution log
    - tool_call_summary.json: Tool usage summary for each turn (optional)
    - tool_call_summary.md: Human-readable tool usage summary (optional)
"""

import argparse
import asyncio
import glob
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import yaml

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.runners.folder_task import run_folder_task_simple, get_results_dir_for_folder
from src.llm.exceptions import APIConnectionSkipError, APIRateLimitError, APITimeoutError
from src.llm.error_logger import log_api_error, log_api_skip, get_log_file_path, LOG_DIR

def prepare_folder_for_context_size(folder_path: str, context_size: str, embedding_type: str = "gpt") -> str:
    """
    Prepare a folder to use a specific context size.
    
    IMPORTANT: This function NO LONGER renames files to support concurrent execution.
    Instead, it returns the target db path which is passed to run_folder_task_simple.
    The target_db_path is then used by folder_processor.py to filter files and
    by RAG tools to restrict access to only the specified db file.
    
    Args:
        folder_path: Path to the folder containing db files
        context_size: Context size (e.g., "32k", "64k", "128k", "supporting_only", "no_supporting", "no_rag")
        embedding_type: Embedding type to use ("gpt", "qwen", or "bm25"). Default is "gpt".
                       - "gpt": uses long_context_sampled_{size}.json.chunks.db
                       - "qwen": uses long_context_sampled_{size}.json.qwen.chunks.db
                       - "bm25": uses long_context_sampled_{size}.json.bm25.pkl
        
    Returns:
        Path to the target db file, or None if not found (or "NO_RAG" for no_rag mode)
    """
    # Special handling for ablation experiment modes
    if context_size == "no_rag":
        # Return special marker to indicate no RAG should be used
        return "NO_RAG"
    
    if context_size == "online_search":
        # Return special marker for online search ablation experiment
        # This mode uses real-time web search instead of local RAG
        # Unlike no_rag, this does NOT auto-add agent config - user must specify their own
        return "ONLINE_SEARCH"
    
    # Determine the db suffix based on embedding type
    if embedding_type == "qwen":
        db_suffix = ".qwen.chunks.db"
    elif embedding_type == "bm25":
        db_suffix = ".bm25.pkl"
    else:
        db_suffix = ".chunks.db"
    
    if context_size == "supporting_only":
        # Use the supporting_only ablation file
        target_json = os.path.join(folder_path, "long_context_supporting_only.json")
        target_db = target_json + db_suffix
        if os.path.exists(target_db):
            return target_db
        elif os.path.exists(target_json):
            # DB file doesn't exist yet, need to generate it
            print(f"  Note: DB file not found for supporting_only, will use JSON: {target_json}")
            return target_json
        else:
            print(f"  Warning: Ablation file not found: {target_json}")
            print(f"  Please run generate_ablation_contexts.py first")
            return None
    
    if context_size == "no_supporting":
        # Use the no_supporting ablation file
        target_json = os.path.join(folder_path, "long_context_no_supporting.json")
        target_db = target_json + db_suffix
        if os.path.exists(target_db):
            return target_db
        elif os.path.exists(target_json):
            # DB file doesn't exist yet, need to generate it
            print(f"  Note: DB file not found for no_supporting, will use JSON: {target_json}")
            return target_json
        else:
            print(f"  Warning: Ablation file not found: {target_json}")
            print(f"  Please run generate_ablation_contexts.py first")
            return None
    
    if context_size == "no_distractor_128k":
        # Use the no_distractor_128k ablation file (128k context without distractor documents)
        target_json = os.path.join(folder_path, "long_context_no_distractor_128k.json")
        target_db = target_json + db_suffix
        if os.path.exists(target_db):
            return target_db
        elif os.path.exists(target_json):
            # DB file doesn't exist yet, need to generate it
            print(f"  Note: DB file not found for no_distractor_128k, will use JSON: {target_json}")
            return target_json
        else:
            print(f"  Warning: Ablation file not found: {target_json}")
            print(f"  Please run build_long_context_no_distractor_v2.py first")
            return None
    
    # Standard context size handling (32k, 64k, 128k, etc.)
    # Use the appropriate glob pattern based on embedding type
    if embedding_type == "qwen":
        db_files = glob.glob(os.path.join(folder_path, "*.qwen.chunks.db"))
        if not db_files:
            print(f"  Warning: No .qwen.chunks.db files found in {folder_path}")
            return None
        target_pattern = f"long_context_sampled_{context_size}.json.qwen.chunks.db"
    elif embedding_type == "bm25":
        db_files = glob.glob(os.path.join(folder_path, "*.bm25.pkl"))
        if not db_files:
            print(f"  Warning: No .bm25.pkl files found in {folder_path}")
            return None
        target_pattern = f"long_context_sampled_{context_size}.json.bm25.pkl"
    else:
        db_files = glob.glob(os.path.join(folder_path, "*.chunks.db"))
        # Filter out qwen embedding files when looking for gpt embedding
        db_files = [f for f in db_files if not f.endswith(".qwen.chunks.db")]
        if not db_files:
            print(f"  Warning: No .chunks.db files found in {folder_path}")
            return None
        target_pattern = f"long_context_sampled_{context_size}.json.chunks.db"
    target_db = None
    
    for db_file in db_files:
        if os.path.basename(db_file) == target_pattern:
            target_db = db_file
            break
    
    if not target_db:
        print(f"  Warning: No db file found for context size {context_size} in {folder_path}")
        print(f"  Available db files: {[os.path.basename(f) for f in db_files]}")
        return None
    
    # No longer rename files - just return the target path
    # This allows concurrent execution with different context sizes
    return target_db

def restore_folder_db_files(folder_path: str) -> None:
    """
    Restore all hidden db files in a folder.
    
    NOTE: This function is kept for backward compatibility but is no longer needed
    since prepare_folder_for_context_size no longer renames files.
    """
    hidden_files = glob.glob(os.path.join(folder_path, "*.chunks.db.hidden"))
    
    for hidden_file in hidden_files:
        original_path = hidden_file[:-7]
        if not os.path.exists(original_path):
            os.rename(hidden_file, original_path)

def load_tasks_from_jsonl(jsonl_path: str) -> List[Dict]:
    """Load tasks from a JSONL file."""
    tasks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                task = json.loads(line)
                if 'task' in task and 'number' not in task:
                    task['number'] = task['task']
                if 'number' in task and 'query' in task:
                    tasks.append(task)
                else:
                    print(f"Warning: Line {line_num} missing 'number'/'task' or 'query' field, skipping")
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} is not valid JSON: {e}, skipping")
    return tasks

# Minimum report length threshold (in characters)
# Reports shorter than this are considered incomplete and will be regenerated
MIN_REPORT_LENGTH = 500  # About 250 Chinese characters or 100 English words

def is_report_valid(report_path: str, min_length: int = MIN_REPORT_LENGTH) -> bool:
    """
    Check if a report file is valid (exists and has sufficient content).
    
    A report is considered invalid if:
    - It doesn't exist
    - It's too short (less than min_length characters)
    - It only contains headers/metadata without actual content
    
    Args:
        report_path: Path to the report file
        min_length: Minimum content length in characters
        
    Returns:
        True if the report is valid, False otherwise
    """
    if not os.path.exists(report_path):
        return False
    
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove common headers and metadata to get actual content
        # Skip lines starting with # (headers) or containing only metadata
        lines = content.split('\n')
        content_lines = []
        in_report_section = False
        
        for line in lines:
            stripped = line.strip()
            # Skip empty lines and header lines
            if not stripped:
                continue
            if stripped.startswith('#'):
                # Check if we're entering the Report section
                if 'report' in stripped.lower():
                    in_report_section = True
                continue
            if stripped.startswith('Generated at:'):
                continue
            # Only count content after the Report section header
            if in_report_section:
                content_lines.append(stripped)
        
        actual_content = '\n'.join(content_lines)
        content_length = len(actual_content)
        
        # Check if content is too short
        if content_length < min_length:
            return False
        
        return True
        
    except Exception as e:
        print(f"  Warning: Failed to read report {report_path}: {e}")
        return False

def get_completed_tasks(batch_folder: str = None, context_size: str = None, model: str = None, dataset: str = None, run_batch: str = None) -> set:
    """Get set of task numbers that have already been completed.
    
    Directory structure: result/<run_batch>/<dataset>/<context_size>/<model>/<task_number>/
    A task is considered completed if the folder contains a valid final_report.md file.
    
    A report is considered valid if:
    - It exists
    - It has sufficient content (more than MIN_REPORT_LENGTH characters)
    - It doesn't have repetitive/garbage content
    
    Args:
        batch_folder: Full batch folder path (if already constructed)
        context_size: Context size (e.g., "32k")
        model: Model name
        dataset: Dataset name
        run_batch: Run batch identifier.
    """
    completed = set()
    invalid_reports = []
    
    # Build the batch folder path if not provided
    if batch_folder is None and run_batch:
        parts = ["result", run_batch]
        if dataset:
            parts.append(dataset)
        if context_size:
            parts.append(context_size)
        if model:
            parts.append(model)
        batch_folder = os.path.join(*parts)
    
    if batch_folder is None or not os.path.exists(batch_folder):
        return completed
    
    # Check all task folders in the batch folder
    for item in os.listdir(batch_folder):
        item_path = os.path.join(batch_folder, item)
        if os.path.isdir(item_path):
            final_report_path = os.path.join(item_path, "final_report.md")
            if os.path.exists(final_report_path):
                # Check if the report is valid (has sufficient content)
                if is_report_valid(final_report_path):
                    # Task number is the folder name (e.g., "001")
                    completed.add(item)
                else:
                    invalid_reports.append(item)
    
    # Report invalid reports that will be regenerated
    if invalid_reports:
        print(f"⚠ Found {len(invalid_reports)} tasks with invalid/short reports, will regenerate: {invalid_reports[:10]}{'...' if len(invalid_reports) > 10 else ''}")
    
    return completed

def create_run_folder(results_dir: str, task_number: str, context_size: str = None, model: str = None, dataset: str = None, run_batch: str = None) -> str:
    """Create a unique folder for this task run.
    
    Directory structure: result/<run_batch>/<dataset>/<context_size>/<model>/<task_number>/
    Example: result/20251222_220000/datasets_batch2/32k/gpt4.1/001/
    
    Args:
        results_dir: Base results directory (ignored in new structure, uses current directory)
        task_number: Task number (e.g., "001")
        context_size: Context size (e.g., "32k")
        model: Model name (e.g., "gpt4.1")
        dataset: Dataset name (e.g., "datasets_batch2")
        run_batch: Run batch identifier in datetime format (e.g., "20251210_150000").
                   This groups all tasks from the same batch run together.
    """
    # Build folder path: result/<run_batch>/<dataset>/<context_size>/<model>/<task_number>/
    parts = ["result"]
    
    if run_batch:
        parts.append(run_batch)
    else:
        parts.append(datetime.now().strftime('%Y%m%d_%H%M%S'))
    
    if dataset:
        parts.append(dataset)
    
    if context_size:
        parts.append(context_size)
    
    if model:
        parts.append(model)
    
    parts.append(task_number)
    
    # Join with path separator to create the folder path
    run_folder = os.path.join(*parts)
    os.makedirs(run_folder, exist_ok=True)
    return run_folder

def parse_tool_calls_from_log(log_file_path: str) -> Dict:
    """Parse tool call information from the JSON log file."""
    if not os.path.exists(log_file_path):
        return {"error": f"Log file not found: {log_file_path}"}
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
    except Exception as e:
        return {"error": f"Failed to parse log file: {e}"}
    
    tool_calls_by_turn = {}
    current_turn = 0
    
    step_logs = log_data.get('step_logs', [])
    
    for step in step_logs:
        step_name = step.get('step_name', '')
        message = step.get('message', '')
        metadata = step.get('metadata', {})
        
        # Detect turn changes
        if "Turn:" in step_name:
            turn_match = re.search(r'Turn:\s*(\d+)', step_name)
            if turn_match:
                current_turn = int(turn_match.group(1))
                if current_turn not in tool_calls_by_turn:
                    tool_calls_by_turn[current_turn] = {
                        "tools_used": [],
                        "rag_queries": []
                    }
        
        # Detect tool calls
        if "Tool Call Start" in step_name or "Tool Call Success" in step_name:
            tool_info = {}
            
            if "tool-" in step_name.lower():
                tool_match = re.search(r'tool-(\w+)', step_name.lower())
                if tool_match:
                    tool_info["tool_name"] = tool_match.group(1)
            
            if "tool_name" not in tool_info:
                if "rag_search" in message.lower() or "rag_get_context" in message.lower():
                    tool_info["tool_name"] = "rag"
                elif "google" in message.lower() or "search" in message.lower():
                    tool_info["tool_name"] = "google_search"
                elif "browser" in message.lower() or "playwright" in message.lower():
                    tool_info["tool_name"] = "browser"
                elif "python" in message.lower():
                    tool_info["tool_name"] = "python"
                else:
                    if metadata.get('tool_name'):
                        tool_info["tool_name"] = metadata.get('tool_name')
                    elif metadata.get('server_name'):
                        tool_info["tool_name"] = metadata.get('server_name')
            
            if tool_info.get("tool_name") and current_turn in tool_calls_by_turn:
                existing_tools = [t.get("tool_name") for t in tool_calls_by_turn[current_turn]["tools_used"]]
                if tool_info["tool_name"] not in existing_tools or "Tool Call Start" in step_name:
                    tool_calls_by_turn[current_turn]["tools_used"].append(tool_info)
    
    return {
        "task_id": log_data.get('task_id', 'unknown'),
        "status": log_data.get('status', 'unknown'),
        "start_time": log_data.get('start_time', ''),
        "end_time": log_data.get('end_time', ''),
        "total_turns": len(tool_calls_by_turn),
        "turns": tool_calls_by_turn
    }

def parse_rag_from_execution_log(log_file_path: str) -> Dict:
    """
    Parse RAG retrieval information directly from the execution log.
    
    This extracts RAG tool calls and their results from the main execution log,
    since RAG tools don't save separate log files.
    
    Returns:
        Dict containing:
        - total_queries: Number of RAG queries
        - queries: List of query details (with turn and agent info)
        - queries_by_turn: Dict mapping turn number to list of queries
        - queries_by_agent: Dict mapping agent name to list of queries
        - db_paths: Set of unique db file paths used (extracted from json_path arguments)
    """
    if not os.path.exists(log_file_path):
        return {"message": "Execution log not found"}
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
    except Exception as e:
        return {"message": f"Failed to parse execution log: {e}"}
    
    rag_summary = {
        "total_queries": 0,
        "queries": [],
        "queries_by_turn": {},  # {turn_key: [queries]}
        "queries_by_agent": {},  # {agent_name: [queries]}
        "db_paths": set()  # Track unique db file paths used
    }
    
    step_logs = log_data.get('step_logs', [])
    
    # Track RAG tool calls and their results
    current_rag_call = None
    current_turn = 0
    current_agent = "main"  # Default to main agent
    
    for step in step_logs:
        step_name = step.get('step_name', '')
        message = step.get('message', '')
        metadata = step.get('metadata', {})
        timestamp = step.get('timestamp', '')
        
        # Detect agent changes (sub-agent sessions)
        if "Start Task" in step_name:
            # Extract agent name from step_name (e.g., "agent-browsing | Start Task")
            agent_match = re.search(r'(agent-\w+)', step_name)
            if agent_match:
                current_agent = agent_match.group(1)
                current_turn = 0  # Reset turn counter for new agent
        elif "Main Agent" in step_name:
            current_agent = "main"
        
        # Detect turn changes
        if "Turn:" in step_name:
            turn_match = re.search(r'Turn:\s*(\d+)', step_name)
            if turn_match:
                current_turn = int(turn_match.group(1))
        
        # Detect RAG tool call start
        # Note: "tool-rag" is in the message field, not in step_name
        if "Tool Call Start" in step_name and "tool-rag" in message.lower():
            # Extract query and json_path from metadata or message
            tool_args = metadata.get('arguments', {})
            query = tool_args.get('query', '')
            json_path = tool_args.get('json_path', '')
            
            # Track the db path used
            if json_path:
                rag_summary["db_paths"].add(json_path)
            
            # Extract tool name from message (e.g., "to call tool 'rag_search'")
            tool_name_match = re.search(r"to call tool '(\w+)'", message)
            tool_name = tool_name_match.group(1) if tool_name_match else metadata.get('tool_name', 'rag_search')
            
            current_rag_call = {
                "query": query,
                "tool": tool_name,
                "json_path": json_path,
                "turn": current_turn,
                "agent": current_agent,
                "timestamp": timestamp,
                "num_results": 0,
                "retrieved_documents": []
            }
        
        # Detect RAG tool call success with results
        # Note: "tool-rag" is in the message field, not in step_name
        elif "Tool Call Success" in step_name and "tool-rag" in message.lower():
            if current_rag_call:
                # Try to extract results from the tool response
                tool_result = metadata.get('result', '')
                
                # Parse the result to extract document titles
                if isinstance(tool_result, str):
                    # Try to extract document titles from the result text
                    docs = extract_doc_titles_from_result(tool_result)
                    current_rag_call["retrieved_documents"] = docs
                    current_rag_call["num_results"] = len(docs)
                elif isinstance(tool_result, list):
                    for item in tool_result:
                        if isinstance(item, dict):
                            doc_info = {
                                "title": item.get('title', 'Untitled'),
                                "score": item.get('score', 0),
                                "doc_index": item.get('doc_index', 0),
                                "chunk_index": item.get('chunk_index', 0)
                            }
                            current_rag_call["retrieved_documents"].append(doc_info)
                    current_rag_call["num_results"] = len(current_rag_call["retrieved_documents"])
                
                # Add to main queries list
                rag_summary["queries"].append(current_rag_call)
                rag_summary["total_queries"] += 1
                
                # Add to queries_by_turn
                turn_key = f"{current_agent}_turn_{current_turn}"
                if turn_key not in rag_summary["queries_by_turn"]:
                    rag_summary["queries_by_turn"][turn_key] = []
                rag_summary["queries_by_turn"][turn_key].append(current_rag_call)
                
                # Add to queries_by_agent
                if current_agent not in rag_summary["queries_by_agent"]:
                    rag_summary["queries_by_agent"][current_agent] = []
                rag_summary["queries_by_agent"][current_agent].append(current_rag_call)
                
                current_rag_call = None
        
        # Also check for RAG-related log messages that might contain query info
        elif "rag_search" in message.lower() or "rag_get_context" in message.lower():
            # Try to extract query from the message
            query_match = re.search(r'query["\s:]+([^"]+)"', message)
            if query_match and not current_rag_call:
                current_rag_call = {
                    "query": query_match.group(1),
                    "tool": "rag_search",
                    "json_path": "",
                    "turn": current_turn,
                    "agent": current_agent,
                    "timestamp": timestamp,
                    "num_results": 0,
                    "retrieved_documents": []
                }
        
        # Check for diverse search results in log messages
        elif "Diverse search returned" in message:
            match = re.search(r'(\d+) results from (\d+) documents', message)
            if match and current_rag_call:
                current_rag_call["num_results"] = int(match.group(1))
    
    return rag_summary

def extract_doc_titles_from_result(result_text: str) -> List[Dict]:
    """Extract document titles from RAG result text."""
    docs = []
    
    # Pattern 1: Look for "Title: xxx" or "title: xxx" patterns
    title_matches = re.findall(r'[Tt]itle[:\s]+([^\n]+)', result_text)
    for title in title_matches:
        title = title.strip().strip('"\'')
        if title and len(title) > 3:
            docs.append({"title": title, "score": 0})
    
    # Pattern 2: Look for "## Title" markdown headers
    header_matches = re.findall(r'##\s+([^\n]+)', result_text)
    for title in header_matches:
        title = title.strip()
        if title and len(title) > 3 and title not in [d["title"] for d in docs]:
            docs.append({"title": title, "score": 0})
    
    # Pattern 3: Look for "Document X:" patterns
    doc_matches = re.findall(r'Document\s+\d+[:\s]+([^\n]+)', result_text)
    for title in doc_matches:
        title = title.strip().strip('"\'')
        if title and len(title) > 3 and title not in [d["title"] for d in docs]:
            docs.append({"title": title, "score": 0})
    
    # Pattern 4: Look for score patterns like "(score: 0.85)"
    score_matches = re.findall(r'([^\n]+)\s*\(score:\s*([\d.]+)\)', result_text)
    for title, score in score_matches:
        title = title.strip().strip('"\'')
        if title and len(title) > 3:
            # Update existing or add new
            found = False
            for doc in docs:
                if doc["title"] == title:
                    doc["score"] = float(score)
                    found = True
                    break
            if not found:
                docs.append({"title": title, "score": float(score)})
    
    return docs

def save_tool_call_summary(run_folder: str, log_file_path: str, task_id: str, log_dir: str) -> tuple:
    """
    Save tool call summary to the run folder.
    
    Returns:
        Tuple of (json_path, md_path)
    """
    # Parse tool calls from main log
    tool_calls = parse_tool_calls_from_log(log_file_path)
    
    # Parse RAG retrieval information from execution log
    rag_summary = parse_rag_from_execution_log(log_file_path)
    
    # Combine into comprehensive summary
    summary = {
        "task_id": task_id,
        "execution_info": {
            "status": tool_calls.get("status", "unknown"),
            "start_time": tool_calls.get("start_time", ""),
            "end_time": tool_calls.get("end_time", ""),
            "total_turns": tool_calls.get("total_turns", 0)
        },
        "tool_calls_by_turn": tool_calls.get("turns", {}),
        "rag_retrieval": rag_summary
    }
    
    # Save JSON summary
    json_path = os.path.join(run_folder, "tool_call_summary.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Generate human-readable markdown summary
    md_path = os.path.join(run_folder, "tool_call_summary.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Tool Call Summary\n\n")
        f.write(f"**Task ID:** {task_id}\n\n")
        f.write(f"## Execution Info\n\n")
        f.write(f"- **Status:** {summary['execution_info']['status']}\n")
        f.write(f"- **Start Time:** {summary['execution_info']['start_time']}\n")
        f.write(f"- **End Time:** {summary['execution_info']['end_time']}\n")
        f.write(f"- **Total Turns:** {summary['execution_info']['total_turns']}\n\n")
        
        f.write(f"## Tool Calls by Turn\n\n")
        turns = summary.get("tool_calls_by_turn", {})
        for turn_num in sorted(turns.keys(), key=lambda x: int(x) if isinstance(x, str) else x):
            turn_data = turns[turn_num]
            f.write(f"### Turn {turn_num}\n\n")
            
            tools = turn_data.get("tools_used", [])
            if tools:
                f.write("**Tools Used:**\n")
                for tool in tools:
                    f.write(f"- {tool.get('tool_name', 'unknown')}\n")
            else:
                f.write("*No tools used in this turn*\n")
            f.write("\n")
        
        f.write(f"## RAG Retrieval Summary\n\n")
        rag = summary.get("rag_retrieval", {})
        if rag.get("message"):
            f.write(f"*{rag['message']}*\n\n")
        else:
            f.write(f"**Total RAG Queries:** {rag.get('total_queries', 0)}\n\n")
            
            for i, query in enumerate(rag.get("queries", []), 1):
                f.write(f"### RAG Query {i}\n\n")
                f.write(f"**Query:** {query.get('query', '')}\n\n")
                f.write(f"**Tool:** {query.get('tool', '')}\n\n")
                f.write(f"**Results:** {query.get('num_results', 0)}\n\n")
                
                docs = query.get("retrieved_documents", [])
                if docs:
                    f.write("**Retrieved Documents:**\n\n")
                    for doc in docs:
                        title = doc.get('title', 'Untitled')
                        url = doc.get('url', '')
                        score = doc.get('score', 0)
                        f.write(f"- **{title}**\n")
                        if url:
                            f.write(f"  - URL: {url}\n")
                        f.write(f"  - Score: {score:.4f}\n")
                f.write("\n")
    
    return json_path, md_path

def save_rag_queries_summary(run_folder: str, rag_summary: Dict) -> str:
    """
    Save RAG queries summary to a human-readable txt file.
    
    This file shows all RAG queries organized by agent and turn,
    making it easy to see what search terms were used in each round.
    
    Args:
        run_folder: Directory to save the file
        rag_summary: RAG summary dict from parse_rag_from_execution_log
        
    Returns:
        Path to the saved file
    """
    txt_path = os.path.join(run_folder, "rag_queries.txt")
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RAG QUERIES SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        total_queries = rag_summary.get("total_queries", 0)
        f.write(f"Total Queries: {total_queries}\n\n")
        
        # Show db paths used
        db_paths = rag_summary.get("db_paths", set())
        if db_paths:
            f.write("Database Files Used:\n")
            for db_path in sorted(db_paths):
                f.write(f"  - {os.path.basename(db_path)}\n")
            f.write("\n")
        
        f.write("-" * 70 + "\n")
        f.write("RAG Queries by Agent and Turn:\n")
        f.write("-" * 70 + "\n\n")
        
        # Group by agent first
        queries_by_agent = rag_summary.get("queries_by_agent", {})
        
        if not queries_by_agent:
            # Fallback to flat list
            queries = rag_summary.get("queries", [])
            if queries:
                for i, q in enumerate(queries, 1):
                    f.write(f"[Query {i}]\n")
                    f.write(f"  Query: {q.get('query', 'N/A')}\n")
                    f.write(f"  Tool: {q.get('tool', 'N/A')}\n")
                    f.write(f"  Results: {q.get('num_results', 0)}\n")
                    f.write("\n")
            else:
                f.write("No RAG queries found\n")
        else:
            for agent_name in sorted(queries_by_agent.keys()):
                agent_queries = queries_by_agent[agent_name]
                f.write(f"\n{'='*50}\n")
                f.write(f"Agent: {agent_name}\n")
                f.write(f"{'='*50}\n\n")
                
                # Group by turn within agent
                turns = {}
                for q in agent_queries:
                    turn = q.get("turn", 0)
                    if turn not in turns:
                        turns[turn] = []
                    turns[turn].append(q)
                
                for turn_num in sorted(turns.keys()):
                    turn_queries = turns[turn_num]
                    f.write(f"  Turn {turn_num}:\n")
                    f.write(f"  {'-'*40}\n")
                    
                    for i, q in enumerate(turn_queries, 1):
                        query_text = q.get('query', 'N/A')
                        tool = q.get('tool', 'rag_search')
                        num_results = q.get('num_results', 0)
                        
                        f.write(f"    [{i}] Query: {query_text}\n")
                        f.write(f"        Tool: {tool}\n")
                        f.write(f"        Results: {num_results}\n")
                        
                        # Show retrieved document titles (first 3)
                        docs = q.get('retrieved_documents', [])
                        if docs:
                            f.write(f"        Retrieved documents:\n")
                            for j, doc in enumerate(docs[:3], 1):
                                title = doc.get('title', 'Untitled')[:50]
                                score = doc.get('score', 0)
                                f.write(f"          {j}. {title}... (score: {score:.3f})\n")
                            if len(docs) > 3:
                                f.write(f"          ... and {len(docs) - 3} more documents\n")
                        f.write("\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF RAG QUERIES SUMMARY\n")
        f.write("=" * 70 + "\n")
    
    return txt_path

def save_initial_report(run_folder: str, original_report: str, query: str) -> str:
    """Save the initial report (before validation) to the run folder."""
    report_path = os.path.join(run_folder, "initial_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Initial Report (Before Validation)\n\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        if query:
            f.write("## Query\n\n")
            f.write(f"{query}\n\n")
        f.write("## Report\n\n")
        f.write(original_report if original_report else "No report generated.")
    return report_path

def save_final_report(run_folder: str, final_report: str, query: str) -> str:
    """Save the final report (after validation) to the run folder."""
    report_path = os.path.join(run_folder, "final_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Final Report (After Validation)\n\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        if query:
            f.write("## Query\n\n")
            f.write(f"{query}\n\n")
        f.write("## Report\n\n")
        f.write(final_report if final_report else "No report generated.")
    return report_path

async def run_single_task(
    data_dir: str,
    task: Dict,
    results_dir: str,
    config_overrides: List[str] = None,
    context_size: str = None,
    model: str = None,
    dataset: str = None,
    run_batch: str = None,
    embedding_type: str = "gpt"
) -> str:
    """Run a single task and save results to a unique run folder.
    
    Args:
        data_dir: Directory containing task data
        task: Task dictionary with 'number' and 'query' keys
        results_dir: Base directory for results
        config_overrides: List of config overrides
        context_size: Context size (32k, 64k, 128k)
        model: Model name
        dataset: Dataset name (e.g., "datasets_batch2")
        run_batch: Run batch identifier (datetime format, e.g., "20251210_150000")
        embedding_type: Embedding type to use ("gpt" or "qwen"). Default is "gpt".
        
    Returns:
        str: "success", "failed", or "skipped" (for API connection errors)
    """
    task_number = task['number']
    query = task['query']
    folder_path = os.path.join(data_dir, task_number)
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found: {folder_path}, skipping task {task_number}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Running Task {task_number}")
    print(f"Folder: {folder_path}")
    if context_size:
        print(f"Context Size: {context_size}")
    if run_batch:
        print(f"Run Batch: {run_batch}")
    print(f"Query: {query[:100]}...")
    print(f"{'='*60}\n")
    
    # Get target db path for specific context size if specified
    # This is now concurrent-safe - no file renaming is done
    target_db_path = None
    if context_size:
        target_db_path = prepare_folder_for_context_size(folder_path, context_size, embedding_type)
        if not target_db_path:
            print(f"Warning: Could not find db file for context size {context_size}, skipping task {task_number}")
            return False
        
        # Handle special ablation modes
        if target_db_path == "NO_RAG":
            print(f"🚫 NO_RAG mode: RAG tools will be disabled, testing model's own knowledge")
            # For no_rag mode, we need to use a special agent config that disables RAG
            if config_overrides is None:
                config_overrides = []
            config_overrides.append("agent=evaluation_no_rag")
            target_db_path = None  # No db file needed
        else:
            print(f"Using target db file: {os.path.basename(target_db_path)}")
    
    try:
        start_time = time.time()
        
        # Create unique run folder for this task execution
        run_folder = create_run_folder(results_dir, task_number, context_size, model, dataset, run_batch)
        print(f"Results will be saved to: {run_folder}")
        
        result = await run_folder_task_simple(
            folder_path=folder_path,
            query=query,
            config_overrides=config_overrides,
            target_db_path=target_db_path
        )
        
        # run_folder_task_simple now returns 5 values (including statistics_summary)
        final_summary, final_boxed_answer, original_boxed_answer, log_file_path, statistics_summary = result
        
        elapsed_time = time.time() - start_time
        
        # Check if the final answer indicates an error (no valid answer generated)
        # If so, delete the run folder and return "failed" to avoid polluting evaluation results
        if final_boxed_answer and final_boxed_answer.strip() == "No final answer generated.":
            print(f"\n⚠ Task {task_number} produced no valid answer: 'No final answer generated.'")
            print(f"  Deleting run folder to avoid polluting evaluation results...")
            if os.path.exists(run_folder):
                shutil.rmtree(run_folder)
                print(f"  Deleted: {run_folder}")
            return "failed"
        
        # Save initial report (before validation)
        initial_report_path = save_initial_report(run_folder, original_boxed_answer, query)
        print(f"Initial report saved to: {initial_report_path}")
        
        # Save final report (after validation)
        final_report_path = save_final_report(run_folder, final_boxed_answer, query)
        print(f"Final report saved to: {final_report_path}")
        
        # Copy the original log file to the run folder
        log_copy_path = None
        if log_file_path and os.path.exists(log_file_path):
            log_copy_path = os.path.join(run_folder, "execution_log.json")
            shutil.copy2(log_file_path, log_copy_path)
            print(f"Execution log copied to: {log_copy_path}")
        
        # Parse RAG usage from execution log and save to txt file
        if log_copy_path and os.path.exists(log_copy_path):
            rag_summary = parse_rag_from_execution_log(log_copy_path)
            if rag_summary.get("total_queries", 0) > 0:
                rag_queries_path = save_rag_queries_summary(run_folder, rag_summary)
                print(f"RAG queries summary saved to: {rag_queries_path}")
                # Also print db paths used for verification
                db_paths = rag_summary.get("db_paths", set())
                if db_paths:
                    print(f"RAG databases used: {[os.path.basename(p) for p in db_paths]}")
        
        # Save statistics summary to txt file
        if statistics_summary:
            stats_path = os.path.join(run_folder, "statistics_summary.txt")
            with open(stats_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("TASK EXECUTION STATISTICS\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Task Number: {task_number}\n")
                f.write(f"Execution Time: {elapsed_time:.2f} seconds\n")
                f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("-" * 60 + "\n")
                f.write(statistics_summary)
                f.write("\n" + "=" * 60 + "\n")
            print(f"Statistics summary saved to: {stats_path}")
        
        print(f"\n✓ Task {task_number} completed in {elapsed_time:.1f}s")
        print(f"  Run folder: {run_folder}")
        
        return "success"
    
    except (APIConnectionSkipError, APIRateLimitError, APITimeoutError) as e:
        # API connection/rate limit/timeout error - skip this task immediately without generating any content
        error_type = type(e).__name__
        
        # Determine the error category for display
        if isinstance(e, APIRateLimitError):
            error_category = "API rate limit / TPM exceeded"
        elif isinstance(e, APITimeoutError):
            error_category = "API timeout"
        else:
            error_category = "API connection error"
        
        print(f"\n⚠ Task {task_number} SKIPPED due to {error_category}:")
        print(f"  {e}")
        
        # Extract trace ID and log to separate error log file
        trace_id = e.get_trace_id() if hasattr(e, 'get_trace_id') else None
        
        # Log to separate API error log file
        log_api_error(
            error_type=error_type,
            model_name=model or "unknown",
            task_id=task_number,
            error_message=str(e),
            trace_id=trace_id,
            original_error=e.original_error if hasattr(e, 'original_error') else None,
            extra_info={
                "context_size": context_size or "default",
                "dataset": dataset or "unknown",
                "run_batch": run_batch or "unknown"
            }
        )
        
        # Also log the skip event
        log_api_skip(
            model_name=model or "unknown",
            task_id=task_number,
            reason=error_type,
            trace_id=trace_id,
            extra_info={
                "context_size": context_size or "default",
                "dataset": dataset or "unknown",
                "run_batch": run_batch or "unknown"
            }
        )
        
        if trace_id:
            print(f"  TraceID: {trace_id}")
            print(f"  (TraceID logged to: {get_log_file_path()})")
        
        print(f"  Moving to next task...")
        
        # Clean up the run folder if it was created but is empty
        if 'run_folder' in locals() and os.path.exists(run_folder):
            # Check if folder is empty or only has minimal files
            files_in_folder = os.listdir(run_folder)
            if len(files_in_folder) == 0:
                os.rmdir(run_folder)
                print(f"  Cleaned up empty run folder: {run_folder}")
        
        return "skipped"
        
    except Exception as e:
        print(f"\n✗ Task {task_number} failed: {e}")
        import traceback
        traceback.print_exc()
        return "failed"
    
    finally:
        # Restore hidden db files
        if context_size:
            restore_folder_db_files(folder_path)

def save_batch_summary(
    stats: Dict,
    results_dir: str,
    context_size: str = None,
    model: str = None,
    dataset: str = None,
    run_batch: str = None,
    elapsed_time: float = 0
) -> str:
    """
    Save batch execution summary to a file in the result folder.
    
    This file contains:
    - Execution statistics
    - List of tasks that need to be retried (skipped due to API errors)
    - List of failed tasks
    
    Args:
        stats: Statistics dictionary from run_batch_tasks
        results_dir: Base results directory
        context_size: Context size used
        model: Model name
        dataset: Dataset name
        run_batch: Run batch identifier
        elapsed_time: Total elapsed time in seconds
        
    Returns:
        Path to the saved summary file
    """
    # Build the batch folder path
    parts = ["result"]
    if run_batch:
        parts.append(run_batch)
    if dataset:
        parts.append(dataset)
    if context_size:
        parts.append(context_size)
    if model:
        parts.append(model)
    
    batch_folder = os.path.join(*parts)
    os.makedirs(batch_folder, exist_ok=True)
    
    # Create summary file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_filename = f"BATCH_SUMMARY_{timestamp}.txt"
    summary_path = os.path.join(batch_folder, summary_filename)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("BATCH EXECUTION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Configuration:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Run Batch:    {run_batch or 'N/A'}\n")
        f.write(f"  Dataset:      {dataset or 'N/A'}\n")
        f.write(f"  Context Size: {context_size or 'default'}\n")
        f.write(f"  Model:        {model or 'N/A'}\n")
        f.write("\n")
        
        f.write("Execution Statistics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Total Time:   {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)\n")
        f.write(f"  Total Tasks:  {stats['total']}\n")
        f.write(f"  Successful:   {stats['success']}\n")
        f.write(f"  Failed:       {stats['failed']}\n")
        f.write(f"  Skipped:      {stats['skipped']}\n")
        f.write(f"    - Already completed: {stats.get('skipped_completed', 0)}\n")
        f.write(f"    - API errors:        {stats.get('skipped_connection', 0)}\n")
        f.write("\n")
        
        # Tasks that need to be retried
        skipped_tasks = stats.get('skipped_connection_tasks', [])
        if skipped_tasks:
            f.write("=" * 70 + "\n")
            f.write("⚠️  TASKS TO RETRY (Skipped due to API errors)\n")
            f.write("=" * 70 + "\n\n")
            f.write("The following tasks were skipped due to API connection/rate limit errors.\n")
            f.write("You can retry them by running:\n\n")
            
            # Generate retry command
            retry_tasks = " ".join(skipped_tasks)
            cmd_parts = ["uv run python run_batch_folder_tasks.py"]
            cmd_parts.append(f"--data-dir {dataset}")
            if context_size:
                cmd_parts.append(f"--context-size {context_size}")
            if model:
                cmd_parts.append(f"--model {model}")
            if run_batch:
                cmd_parts.append(f"--run-batch {run_batch}")
            cmd_parts.append(f"--tasks {retry_tasks}")
            
            f.write(f"  {' '.join(cmd_parts)}\n\n")
            
            f.write("Tasks to retry:\n")
            for task_id in skipped_tasks:
                f.write(f"  - {task_id}\n")
            f.write("\n")
        
        # Failed tasks
        failed_tasks = stats.get('failed_tasks', [])
        if failed_tasks:
            f.write("=" * 70 + "\n")
            f.write("❌ FAILED TASKS\n")
            f.write("=" * 70 + "\n\n")
            f.write("The following tasks failed with errors:\n")
            for task_id in failed_tasks:
                f.write(f"  - {task_id}\n")
            f.write("\n")
        
        # Success message if all tasks completed
        if not skipped_tasks and not failed_tasks:
            f.write("=" * 70 + "\n")
            f.write("✅ ALL TASKS COMPLETED SUCCESSFULLY\n")
            f.write("=" * 70 + "\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF BATCH SUMMARY\n")
        f.write("=" * 70 + "\n")
    
    return summary_path

async def run_batch_tasks(
    data_dir: str,
    tasks: List[Dict],
    results_dir: str,
    skip_completed: bool = False,
    config_overrides: List[str] = None,
    context_size: str = None,
    model: str = None,
    dataset: str = None,
    run_batch: str = None,
    embedding_type: str = "gpt"
) -> Dict:
    """Run multiple tasks in sequence.
    
    Args:
        data_dir: Directory containing task data
        tasks: List of task dictionaries
        results_dir: Base directory for results
        skip_completed: Whether to skip already completed tasks
        config_overrides: List of config overrides
        context_size: Context size (32k, 64k, 128k)
        model: Model name
        dataset: Dataset name (e.g., "datasets_batch2")
        run_batch: Run batch identifier. If not provided, generates one based on current datetime.
        embedding_type: Embedding type to use ("gpt" or "qwen"). Default is "gpt".
    """
    # Generate run_batch timestamp if not provided
    if run_batch is None:
        run_batch = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"📦 Run Batch: {run_batch}")
    if dataset:
        print(f"📂 Dataset: {dataset}")
    
    # Always check completed tasks within the same run batch (auto skip-completed)
    # This ensures we don't re-run tasks that were already completed in this batch
    completed_tasks = get_completed_tasks(
        context_size=context_size,
        model=model,
        dataset=dataset,
        run_batch=run_batch
    )
    
    if completed_tasks:
        print(f"🔄 Found {len(completed_tasks)} completed tasks in this batch, will skip them")
    
    stats = {
        'total': len(tasks),
        'skipped': 0,
        'skipped_completed': 0,  # Tasks skipped because already completed
        'skipped_connection': 0,  # Tasks skipped due to API connection errors
        'success': 0,
        'failed': 0,
        'failed_tasks': [],
        'skipped_connection_tasks': [],  # Track tasks skipped due to connection errors
        'run_batch': run_batch,
        'context_size': context_size,
        'model': model,
        'dataset': dataset
    }
    
    for i, task in enumerate(tasks, 1):
        task_number = task['number']
        
        if task_number in completed_tasks:
            print(f"Skipping task {task_number} (already completed)")
            stats['skipped'] += 1
            stats['skipped_completed'] += 1
            continue
        
        print(f"\n[{i}/{len(tasks)}] Processing task {task_number}...")
        
        result = await run_single_task(
            data_dir=data_dir,
            task=task,
            results_dir=results_dir,
            config_overrides=config_overrides,
            context_size=context_size,
            model=model,
            dataset=dataset,
            run_batch=run_batch,
            embedding_type=embedding_type
        )
        
        if result == "success":
            stats['success'] += 1
        elif result == "skipped":
            stats['skipped'] += 1
            stats['skipped_connection'] += 1
            stats['skipped_connection_tasks'].append(task_number)
        else:  # "failed"
            stats['failed'] += 1
            stats['failed_tasks'].append(task_number)
    
    return stats

def preview_tasks(data_dir: str, tasks: List[Dict], results_dir: str):
    """Preview tasks without running them."""
    completed = get_completed_tasks(results_dir)
    
    print("\n" + "=" * 60)
    print("BATCH TASK PREVIEW")
    print("=" * 60)
    print(f"\nData directory: {data_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Already completed: {len(completed)}")
    print(f"Pending: {len(tasks) - len([t for t in tasks if t['number'] in completed])}")
    
    print("\n" + "-" * 60)
    print("Tasks:")
    print("-" * 60)
    
    for task in tasks:
        task_number = task['number']
        folder_path = os.path.join(data_dir, task_number)
        folder_exists = os.path.exists(folder_path)
        is_completed = task_number in completed
        
        status = "✓ completed" if is_completed else ("✗ folder missing" if not folder_exists else "○ pending")
        
        print(f"\n[{task_number}] {status}")
        print(f"  Folder: {folder_path}")
        print(f"  Query: {task['query'][:80]}...")
    
    print("\n" + "=" * 60)

def get_model_name_from_llm_config(llm_config: str) -> str:
    """
    Read model_name from the LLM config file.
    
    Args:
        llm_config: Name of the LLM config (e.g., 'qwen3_235b' for conf/llm/qwen3_235b.yaml)
    
    Returns:
        The model_name from the config file, or the llm_config name if not found.
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "conf", "llm", f"{llm_config}.yaml")
    
    if not os.path.exists(config_path):
        print(f"Warning: LLM config file not found: {config_path}")
        return llm_config
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config and 'model_name' in config:
            return config['model_name']
        else:
            print(f"Warning: 'model_name' not found in {config_path}, using config name")
            return llm_config
    except Exception as e:
        print(f"Warning: Failed to read LLM config: {e}")
        return llm_config

def main():
    parser = argparse.ArgumentParser(
        description="Run batch folder tasks from a JSONL file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all tasks
    uv run python run_batch_folder_tasks.py --data-dir data/bench_case1104
    
    # Run specific tasks
    uv run python run_batch_folder_tasks.py --data-dir data/bench_case1104 --tasks 001 002
    
    # Skip completed tasks
    uv run python run_batch_folder_tasks.py --data-dir data/bench_case1104 --skip-completed
    
    # Preview without running
    uv run python run_batch_folder_tasks.py --data-dir data/bench_case1104 --preview
    
    # Run with specific context size and LLM config (model name auto-detected from config)
    uv run python run_batch_folder_tasks.py --data-dir datasets --context-size 32k --llm-config qwen3_235b
        """
    )
    
    parser.add_argument(
        "--data-dir", "-d",
        required=True,
        help="Path to the data directory containing task folders and query.jsonl"
    )
    parser.add_argument(
        "--query-file", "-q",
        default="query.jsonl",
        help="Name of the JSONL file containing queries (default: query.jsonl)"
    )
    parser.add_argument(
        "--results-dir", "-r",
        default=None,
        help="Directory to save results (default: results/<data-dir-name>)"
    )
    parser.add_argument(
        "--tasks", "-t",
        nargs="+",
        default=None,
        help="Specific task numbers to run (e.g., 001 002 003)"
    )
    parser.add_argument(
        "--skip-completed", "-s",
        action="store_true",
        help="Skip tasks that have already been completed"
    )
    parser.add_argument(
        "--preview", "-p",
        action="store_true",
        help="Preview tasks without running them"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Offline mode: no web search, only use long context (RAG) as information source"
    )
    parser.add_argument(
        "--context-size", "-c",
        type=str,
        choices=["32k", "64k", "128k", "256k", "512k", "1m", "supporting_only", "no_supporting", "no_rag", "no_distractor_128k"],
        default=None,
        help="Context size to use (32k, 64k, 128k, 256k, 512k, 1m) or ablation mode "
             "(supporting_only, no_supporting, no_rag, no_distractor_128k). Uses the corresponding db file."
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model name for result directory naming. If not provided and --llm-config is specified, "
             "the model_name will be read from the LLM config file."
    )
    parser.add_argument(
        "--llm-config", "-l",
        type=str,
        default=None,
        help="LLM config to use (e.g., 'qwen3_235b' to use conf/llm/qwen3_235b.yaml). "
             "The model_name from this config will be used for result directory naming."
    )
    parser.add_argument(
        "--run-batch", "-b",
        type=str,
        default=None,
        help="Run batch identifier (datetime format, e.g., '20251210_150000'). "
             "If not provided, a new batch ID will be generated. "
             "Use this to continue a previous batch run or to specify a custom batch ID."
    )
    parser.add_argument(
        "--embedding-type", "-e",
        type=str,
        choices=["gpt", "qwen", "bm25"],
        default="gpt",
        help="Embedding type to use for RAG retrieval. "
             "'gpt' uses .chunks.db files (default), "
             "'qwen' uses .qwen.chunks.db files, "
             "'bm25' uses .bm25.pkl files (keyword-based retrieval)."
    )
    parser.add_argument(
        "--ignore-cases", "-i",
        nargs="+",
        default=None,
        help="Case IDs to ignore/skip (e.g., b1_008 b2_037 b2_038 b2_042)"
    )

    args = parser.parse_args()
    
    # Resolve paths
    data_dir = args.data_dir
    query_file = os.path.join(data_dir, args.query_file)
    
    if args.results_dir:
        results_dir = args.results_dir
    else:
        # Use results/results_<context_size>/ format when context_size is specified
        # Example: results/results_32k/qwen2.5-7b-instruct/20251210_150000/001_20251210_150537/
        if args.context_size:
            results_dir = os.path.join("results", f"results_{args.context_size}")
        else:
            data_dir_name = os.path.basename(os.path.abspath(data_dir))
            results_dir = os.path.join("results", f"{data_dir_name}_{args.model}")
    
    # Check query file exists
    if not os.path.exists(query_file):
        print(f"Error: Query file not found: {query_file}")
        sys.exit(1)
    
    # Load tasks
    tasks = load_tasks_from_jsonl(query_file)
    print(f"Loaded {len(tasks)} tasks from {query_file}")
    
    # Filter tasks if specific ones requested
    if args.tasks:
        tasks = [t for t in tasks if t['number'] in args.tasks]
        print(f"Filtered to {len(tasks)} tasks: {args.tasks}")
    
    # Filter out ignored cases
    if args.ignore_cases:
        ignore_set = set(args.ignore_cases)
        original_count = len(tasks)
        tasks = [t for t in tasks if t['number'] not in ignore_set]
        ignored_count = original_count - len(tasks)
        if ignored_count > 0:
            print(f"🚫 Ignoring {ignored_count} cases: {args.ignore_cases}")
    
    if not tasks:
        print("No tasks to process")
        sys.exit(0)
    
    # Prepare config overrides for offline mode and LLM config
    config_overrides = []
    if args.offline:
        config_overrides.append("agent=evaluation_offline")
        print("🔒 Running in OFFLINE mode: No web search, using long context (RAG) only")
    
    # Determine model name: use --model if provided, otherwise read from llm-config
    model_name = args.model
    if args.llm_config:
        config_overrides.append(f"llm={args.llm_config}")
        # If model name not explicitly provided, read from config file
        if model_name is None:
            model_name = get_model_name_from_llm_config(args.llm_config)
        print(f"🤖 Using LLM config: {args.llm_config}")
    
    # Default model name if still not set
    if model_name is None:
        model_name = "gpt4.1"
    
    # Extract dataset name from data_dir
    dataset_name = os.path.basename(os.path.abspath(data_dir))
    
    # Print context size info
    if args.context_size:
        print(f"📊 Using context size: {args.context_size}")
        print(f"🤖 Model: {model_name}")
        print(f"📂 Dataset: {dataset_name}")
        print(f"🔤 Embedding type: {args.embedding_type}")
    
    # Preview or run
    if args.preview:
        preview_tasks(data_dir, tasks, results_dir)
    else:
        print(f"\nStarting batch processing...")
        print(f"Results will be saved to: {results_dir}")
        
        start_time = time.time()
        
        stats = asyncio.run(
            run_batch_tasks(
                data_dir=data_dir,
                tasks=tasks,
                results_dir=results_dir,
                skip_completed=args.skip_completed,
                config_overrides=config_overrides if config_overrides else None,
                context_size=args.context_size,
                model=model_name,
                dataset=dataset_name,
                run_batch=args.run_batch,
                embedding_type=args.embedding_type
            )
        )
        
        elapsed_time = time.time() - start_time
        
        # Save batch summary to file
        summary_path = save_batch_summary(
            stats=stats,
            results_dir=results_dir,
            context_size=args.context_size,
            model=model_name,
            dataset=dataset_name,
            run_batch=stats.get('run_batch'),
            elapsed_time=elapsed_time
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("BATCH PROCESSING COMPLETE")
        print("=" * 60)
        print(f"\nTotal time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")
        print(f"Total tasks: {stats['total']}")
        print(f"Successful: {stats['success']}")
        print(f"Failed: {stats['failed']}")
        print(f"Skipped: {stats['skipped']}")
        
        if stats['failed_tasks']:
            print(f"\nFailed tasks: {', '.join(stats['failed_tasks'])}")
        
        if stats.get('skipped_connection_tasks'):
            print(f"\nSkipped due to API connection errors: {', '.join(stats['skipped_connection_tasks'])}")
            print("  (These tasks can be retried later)")
        
        print(f"\nResults saved to: {results_dir}")
        print(f"Batch summary saved to: {summary_path}")
        print("=" * 60)

if __name__ == "__main__":
    main()
