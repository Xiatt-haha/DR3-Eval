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

Usage:
    python -m evaluators.format_compliance \
        --result examples/001/final_report.md \
        --checklist checklists/batch2_checklists/001/checklist.json \
        --output examples/001/eval_format_compliance.json
    
    python -m evaluators.format_compliance \
        --result result/001/final_report.md \
        --data-dir datasets_batch4 \
        --task 001 \
"""

import json
import logging
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

from .utils.base import BaseEvaluator, EvalConfig, EvalResult
from .utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

DEFAULT_CHECKLISTS_DIR = "evaluation/checklists"

class FormatComplianceEvaluator(BaseEvaluator):
    
    metric_name = "format_compliance"
    weight = 1.0
    
    def __init__(self, config: Optional[EvalConfig] = None):
        super().__init__(config)
        self.llm = LLMClient(self.config)
        self.max_retries = 3
    
    def evaluate(self, result_text: str,
                 checklist_path: Optional[Path] = None,
                 checklist_items: Optional[List[Dict]] = None,
                 query: str = None,
                 data_dir: str = None,
                 task_number: str = None,
                 checklists_base_dir: str = None,
                 auto_generate: bool = True,
                 **kwargs) -> EvalResult:
        items = checklist_items or []
        if checklist_path and checklist_path.exists():
            items = self._load_checklist(checklist_path)
        
        if not items and data_dir and task_number:
            print(f"  No checklist provided, attempting to load or generate...")
            items = self._load_or_generate_checklist(
                data_dir=data_dir,
                task_number=task_number,
                query=query,
                checklists_base_dir=checklists_base_dir,
                auto_generate=auto_generate
            )
        
        if not items:
            error_msg = "No checklist provided"
            logger.error(error_msg)
            print(f"❌ ERROR: {error_msg}")
            return EvalResult(
                metric_name=self.metric_name,
                score=-1,  # Error indicator
                details={
                    'error': error_msg,
                    'status': 'failed'
                },
                weight=self.weight
            )
        
        evaluation_results = self._evaluate_checklist(result_text, items)
        
        error_results = [r for r in evaluation_results if r.get('status') == 'error']
        if len(error_results) == len(evaluation_results):
            error_msg = f"All {len(evaluation_results)} checklist items failed to evaluate"
            logger.error(error_msg)
            print(f"❌ ERROR: {error_msg}")
            return EvalResult(
                metric_name=self.metric_name,
                score=-1,  # Error indicator
                details={
                    'error': error_msg,
                    'status': 'failed',
                    'failed_items': error_results
                },
                weight=self.weight
            )
        
        valid_results = [r for r in evaluation_results if r.get('status') != 'error']
        satisfied_count = sum(1 for r in valid_results if r.get('satisfied'))
        total_count = len(valid_results)
        score = (satisfied_count / total_count * 100) if total_count > 0 else 0.0
        
        category_stats = {}
        for r in evaluation_results:
            cat = r.get('category', 'content')
            if cat not in category_stats:
                category_stats[cat] = {'satisfied': 0, 'total': 0}
            category_stats[cat]['total'] += 1
            if r.get('satisfied'):
                category_stats[cat]['satisfied'] += 1
        
        satisfied_items = [r.get('requirement', '') for r in evaluation_results if r.get('satisfied')]
        unsatisfied_items = [r.get('requirement', '') for r in evaluation_results if not r.get('satisfied')]
        
        details = {
            'method': 'checklist',
            'checklist_evaluation': evaluation_results,
            'satisfied_count': satisfied_count,
            'total_items': total_count,
            'total_count': total_count,
            'satisfaction_rate': score,
            'category_stats': category_stats,
            'satisfied_items': satisfied_items,
            'unsatisfied_items': unsatisfied_items
        }
        
        return EvalResult(
            metric_name=self.metric_name,
            score=score,
            details=details,
            weight=self.weight
        )
    
    def _load_checklist(self, path: Path) -> List[Dict]:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            return data.get('checklist', [])
        except Exception:
            return []
    
    def _evaluate_checklist(self, result_text: str, items: List[Dict]) -> List[Dict]:
        total = len(items)
        
        batch_size = 5
        
        if total <= batch_size:
            print(f"  Evaluating {total} checklist items in one batch...")
            return self._evaluate_checklist_batch(result_text, items, 0, total)
        
        all_results = []
        num_batches = (total + batch_size - 1) // batch_size
        print(f"  Evaluating {total} checklist items in {num_batches} batches (batch size: {batch_size})...")
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total)
            batch_items = items[start_idx:end_idx]
            
            print(f"    Batch {batch_idx + 1}/{num_batches}: items {start_idx + 1}-{end_idx}")
            batch_results = self._evaluate_checklist_batch(result_text, batch_items, start_idx, total)
            all_results.extend(batch_results)
        
        return all_results
    
    def _evaluate_checklist_batch(self, result_text: str, items: List[Dict], start_idx: int, total: int) -> List[Dict]:
        batch_size = len(items)
        
        checklist_list = "\n".join([
            f"{i+1}. {item.get('requirement', '')}"
            for i, item in enumerate(items)
        ])
        
        prompt = f"""Check if the report satisfies each requirement. Output ONLY true/false for each item.

## Requirements ({batch_size} items)
{checklist_list}

## Report
{result_text}

## Rules
- true: Report clearly addresses this requirement with specific content
- false: Not mentioned, or only briefly/superficially mentioned

## Response Format (JSON only, no explanation needed)
{{"results": [{", ".join([f'{{"id": {i+1}, "satisfied": true/false}}' for i in range(batch_size)])}]}}"""
        
        for attempt in range(self.max_retries):
            try:
                result = self.llm.call(
                    system="You are a meticulous requirements checker.",
                    user=prompt
                )
                
                if result and isinstance(result, dict) and 'results' in result:
                    llm_results = result.get('results', [])
                    evaluation_results = []
                    
                    for i, item in enumerate(items):
                        item_id = item.get('id', i+1)
                        requirement = item.get('requirement', '')
                        category = item.get('category', 'content')
                        
                        llm_result = None
                        for r in llm_results:
                            if r.get('id') == i + 1:
                                llm_result = r
                                break
                        
                        if llm_result:
                            satisfied = llm_result.get('satisfied', False)
                            explanation = llm_result.get('explanation', '')
                            status_icon = "✅" if satisfied else "❌"
                            print(f"    [{i+1}/{total}] {status_icon} {requirement[:50]}...")
                            
                            evaluation_results.append({
                                'id': item_id,
                                'requirement': requirement,
                                'category': category,
                                'satisfied': satisfied,
                                'explanation': explanation,
                                'status': 'success'
                            })
                        else:
                            print(f"    [{i+1}/{total}] ⚠️ No result for: {requirement[:50]}...")
                            evaluation_results.append({
                                'id': item_id,
                                'requirement': requirement,
                                'category': category,
                                'satisfied': None,
                                'explanation': 'No result from LLM for this item',
                                'status': 'error'
                            })
                    
                    return evaluation_results
                else:
                    logger.warning(f"Invalid response format (attempt {attempt + 1})")
                    
            except Exception as e:
                logger.warning(f"Error evaluating checklist (attempt {attempt + 1}): {e}")
        
        error_msg = f"Evaluation failed after {self.max_retries} attempts"
        logger.error(error_msg)
        print(f"  ⚠️ ERROR: {error_msg}")
        
        return [{
            'id': item.get('id', i+1),
            'requirement': item.get('requirement', ''),
            'category': item.get('category', 'content'),
            'satisfied': None,
            'explanation': error_msg,
            'status': 'error'
        } for i, item in enumerate(items)]
    
    def _load_or_generate_checklist(
        self,
        data_dir: str,
        task_number: str,
        query: str = None,
        checklists_base_dir: str = None,
        auto_generate: bool = True
    ) -> List[Dict]:
        if checklists_base_dir is None:
            checklists_base_dir = DEFAULT_CHECKLISTS_DIR
        
        dataset_name = os.path.basename(os.path.normpath(data_dir))
        checklist_dir = os.path.join(checklists_base_dir, dataset_name, task_number)
        checklist_file = os.path.join(checklist_dir, "checklist.json")
        
        if os.path.exists(checklist_file):
            print(f"  Loading existing checklist from: {checklist_file}")
            try:
                with open(checklist_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    items = data.get("checklist", [])
                print(f"  Loaded {len(items)} checklist items")
                return items
            except Exception as e:
                print(f"  Warning: Failed to load checklist: {e}")
        
        if auto_generate:
            print(f"  Checklist not found, attempting to generate...")
            
            if not query:
                query = self._load_query_from_file(data_dir, task_number)
            
            if query:
                try:
                    from .utils.generate_checklists import generate_checklist_for_task
                    
                    items = generate_checklist_for_task(
                        data_dir=data_dir,
                        task_number=task_number,
                        query=query,
                        output_dir=checklists_base_dir
                    )
                    return items
                        
                except ImportError as e:
                    print(f"  Warning: Could not import checklist generation utilities: {e}")
                except Exception as e:
                    print(f"  Warning: Failed to generate checklist: {e}")
            else:
                print(f"  Warning: No query provided, cannot generate checklist")
        
        return []
    
    def _load_query_from_file(self, data_dir: str, task_number: str) -> Optional[str]:
        query_file = os.path.join(data_dir, "query.jsonl")
        if not os.path.exists(query_file):
            return None
        
        try:
            with open(query_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        task = json.loads(line)
                        task_num = task.get("task") or task.get("number")
                        if task_num == task_number:
                            return task.get("query")
        except Exception as e:
            print(f"  Warning: Failed to load query from file: {e}")
        
        return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate format compliance using checklist")
    parser.add_argument("--result", type=str, required=True, help="Path to result markdown file")
    parser.add_argument("--checklist", type=str, help="Path to checklist.json file")
    parser.add_argument("--data-dir", type=str, help="Path to data directory (for auto-generation)")
    parser.add_argument("--task", type=str, help="Task number (for auto-generation)")
    parser.add_argument("--query", type=str, help="Query text (for auto-generation)")
    parser.add_argument("--checklists-dir", type=str, help="Path to checklists base directory")
    parser.add_argument("--output", type=str, help="Output file for evaluation result")
    
    args = parser.parse_args()
    
    result_text = Path(args.result).read_text(encoding='utf-8')
    
    evaluator = FormatComplianceEvaluator()
    result = evaluator.evaluate(
        result_text,
        checklist_path=Path(args.checklist) if args.checklist else None,
        query=args.query,
        data_dir=args.data_dir,
        task_number=args.task,
        checklists_base_dir=args.checklists_dir
    )
    
    print(f"\n📋 Format Compliance Score: {result.score:.1f}/100")
    print(result.to_json())
    
    if args.output:
        evaluator.save_result(result, Path(args.output))
        print(f"\n📄 Result saved to: {args.output}")

if __name__ == "__main__":
    main()
