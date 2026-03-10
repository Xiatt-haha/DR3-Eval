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
    python eval.py single \
        --result-dir result/20251227/datasets_batch2/32k/gpt-4.1 \
        --insights-dir insights/batch2_insights
    
    python eval.py single \
        --result-dir result/20251227/datasets_batch2/32k/gpt-4.1 \
        --insights-dir insights/batch2_insights \
        --metrics information_recall overall_quality format_compliance citation_coverage tool_usage
    
    python eval.py all \
        --result-base result/20251227-new/datasets_batch2 \
        --output-dir evaluation_logs/batch_20251227_new
    
    python eval.py all \
        --result-base result/20251227-new/datasets_batch2 \
        --metrics information_recall overall_quality
    
    python eval.py all \
        --result-base result/20251227-new/datasets_batch2 \
        --exclude-models claude37_sonnet
    
    python eval.py all \
        --result-base result/20251227-new/datasets_batch2 \
        --metrics information_recall overall_quality format_compliance citation_coverage tool_usage \
        --workers 15
    
    python eval.py all \
        --result-base result/20251227-new/datasets_batch2 \
        --metrics factual_accuracy \
        --workers 1 \
        --gemini-rpm 10

Available metrics:
"""

import argparse
import json
import logging
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

class GlobalProgressManager:
    """Manages global progress tracking for batch evaluation."""
    
    def __init__(self):
        self.pbar = None
        self.total_cases = 0
        self.completed_cases = 0
        self.current_case = ""
        self._lock = None
        self._start_time = None
    
    def init(self, total_configs: int, total_cases: int = 0):
        import threading
        import time
        self._lock = threading.Lock()
        self.total_cases = total_cases
        self.completed_cases = 0
        self._start_time = time.time()
        
        if HAS_TQDM and total_cases > 0:
            self.pbar = tqdm(
                total=total_cases,
                desc="📊 Evaluating",
                unit="case",
                position=0,
                leave=True,
                ncols=120,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} cases [{elapsed}<{remaining}, {rate_fmt}]'
            )
            print(f"\n🚀 Starting evaluation: {total_configs} configs, {total_cases} cases total\n")
    
    def update_config(self, config_name: str = None, increment: int = 0):
        if self._lock and config_name:
            with self._lock:
                self.current_case = config_name
                if self.pbar:
                    self.pbar.set_postfix_str(config_name, refresh=True)
    
    def update_case(self, case_id: str = None, increment: int = 0):
        if self._lock:
            with self._lock:
                if increment > 0:
                    self.completed_cases += increment
                    if self.pbar:
                        self.pbar.update(increment)
                if case_id and self.pbar:
                    self.pbar.set_postfix_str(case_id, refresh=True)
    
    def close(self):
        import time
        if self.pbar:
            self.pbar.close()
            elapsed = time.time() - self._start_time if self._start_time else 0
            avg_time = elapsed / self.completed_cases if self.completed_cases > 0 else 0
            print(f"\n✅ Completed {self.completed_cases}/{self.total_cases} cases in {elapsed:.1f}s (avg: {avg_time:.2f}s/case)\n")
        self.pbar = None

global_progress = GlobalProgressManager()

import sys
sys.path.insert(0, str(Path(__file__).parent))

from evaluators.utils.run_all import EvaluationRunner, ALL_METRICS

logger = logging.getLogger("batch_eval")

def setup_logging(output_dir: Path = None, level: int = logging.INFO):
    """Configure logging for batch evaluation."""
    logger.setLevel(level)
    
    logger.handlers.clear()
    
    simple_formatter = logging.Formatter(fmt='%(message)s')
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    return logger

def run_batch_evaluation(result_dir: Path, insights_dir: Path, output_dir: Path = None, 
                         checklist_dir: Path = None, datasets_dir: Path = None, context_size: str = "32k",
                         skip_existing: bool = True, metrics: list = None, save_to_result_dir: bool = True):
    """Run batch evaluation across multiple cases."""
    
    incremental_mode = metrics is not None and len(metrics) < len(ALL_METRICS)
    
    case_dirs = sorted([d for d in result_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    
    logger.info(f"Found {len(case_dirs)} cases to evaluate")
    logger.info(f"Skip existing: {skip_existing}")
    logger.info(f"Save to result dir: {save_to_result_dir}")
    if incremental_mode:
        logger.info(f"🔄 Incremental mode: will merge with existing results")
    logger.info(f"Result dir: {result_dir}")
    logger.info(f"Insights dir: {insights_dir}")
    if datasets_dir:
        logger.info(f"Datasets dir: {datasets_dir}")
    if checklist_dir:
        logger.info(f"Checklist dir: {checklist_dir}")
    logger.info(f"Context size: {context_size}")
    if metrics:
        logger.info(f"Metrics: {', '.join(metrics)}")
    else:
        logger.info(f"Metrics: all ({', '.join(ALL_METRICS)})")
    logger.info("=" * 60)
    
    runner = EvaluationRunner(metrics=metrics)
    
    all_results = {}
    summary_scores = []
    
    skipped_existing = 0
    
    for case_dir in case_dirs:
        case_id = case_dir.name
        
        existing_result_path = output_dir / case_id / "combined_result.json" if output_dir else None
        existing_data = None
        
        if existing_result_path and existing_result_path.exists():
            try:
                existing_data = json.loads(existing_result_path.read_text(encoding='utf-8'))
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to load existing result for case {case_id}: {e}")
        
        def extract_metrics_from_existing(existing_data):
            metrics_dict = {}
            for name, m in existing_data.get('metrics', {}).items():
                if name == 'information_recall':
                    details = m.get('details', {})
                    components = details.get('components', {})
                    
                    lc = components.get('long_context', {})
                    if lc.get('available'):
                        metrics_dict['information_recall_longcontext'] = lc.get('score', 0)
                    
                    src = components.get('source_documents', {})
                    if src.get('available'):
                        metrics_dict['information_recall_source'] = src.get('score', 0)
                    
                    metrics_dict['information_recall'] = m.get('score', 0)
                else:
                    metrics_dict[name] = m.get('score', 0)
            return metrics_dict
        
        if incremental_mode and existing_data:
            existing_metrics = set(existing_data.get('metrics', {}).keys())
            requested_metrics = set(metrics)
            
            if skip_existing and requested_metrics.issubset(existing_metrics):
                logger.info(f"  ⏭️ Skipping case {case_id}: all requested metrics already evaluated")
                skipped_existing += 1
                
                all_results[case_id] = existing_data
                summary_scores.append({
                    'case_id': case_id,
                    'total_score': existing_data.get('total_score', 0),
                    'metrics': extract_metrics_from_existing(existing_data)
                })
                continue
            else:
                logger.info(f"  🔄 Case {case_id}: will update metrics {', '.join(requested_metrics)}")
        elif skip_existing and existing_data:
            logger.info(f"  ⏭️ Skipping case {case_id}: already evaluated")
            skipped_existing += 1
            
            all_results[case_id] = existing_data
            summary_scores.append({
                'case_id': case_id,
                'total_score': existing_data.get('total_score', 0),
                'metrics': extract_metrics_from_existing(existing_data)
            })
            continue
        
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Evaluating case {case_id}...")
        logger.info("=" * 60)
        
        final_report = case_dir / "final_report.md"
        execution_log = case_dir / "execution_log.json"
        rag_queries = case_dir / "rag_queries.txt"
        insights_case_dir = insights_dir / case_id
        
        if not final_report.exists():
            logger.warning(f"  ⚠️ Skipping case {case_id}: final_report.md not found")
            continue
        
        gold_path = insights_case_dir / "gold_insights_from_long_context.json" if insights_case_dir.exists() else None
        gold_source_path = insights_case_dir / "gold_insights_from_source.json" if insights_case_dir.exists() else None
        
        long_context_path = None
        useful_search_path = None
        source_folder = None
        if datasets_dir:
            datasets_case_dir = datasets_dir / case_id
            if datasets_case_dir.exists():
                long_context_file = datasets_case_dir / f"long_context_sampled_{context_size}.json"
                if long_context_file.exists():
                    long_context_path = long_context_file
                
                useful_search_file = datasets_case_dir / "useful_search.json"
                useful_search_file_jsonl = datasets_case_dir / "useful_search.jsonl"
                if useful_search_file.exists():
                    useful_search_path = useful_search_file
                elif useful_search_file_jsonl.exists():
                    useful_search_path = useful_search_file_jsonl
                
                source_folder = datasets_case_dir
        
        checklist_path = None
        if checklist_dir:
            checklist_case_dir = checklist_dir / case_id
            if checklist_case_dir.exists():
                checklist_path = checklist_case_dir / "checklist.json"
                if not checklist_path.exists():
                    checklist_path = None
        
        query_file = None
        if datasets_dir:
            query_file = datasets_dir / "query.jsonl"
            if not query_file.exists():
                query_file = None
        
        try:
            result = runner.run(
                result_path=final_report,
                gold_path=gold_path if gold_path and gold_path.exists() else None,
                gold_source_path=gold_source_path if gold_source_path and gold_source_path.exists() else None,
                execution_log_path=execution_log if execution_log.exists() else None,
                checklist_path=checklist_path,
                long_context_path=long_context_path,
                source_folder=source_folder,
                useful_search_path=useful_search_path,
                query_file=query_file,
                case_id=case_id,
                context_size=context_size,
                rag_queries_path=rag_queries if rag_queries.exists() else None
            )
            
            if incremental_mode and existing_data:
                logger.info(f"  🔄 Merging new metrics with existing results...")
                
                merged_metrics = existing_data.get('metrics', {}).copy()
                for name, r in result.results.items():
                    merged_metrics[name] = r.to_dict()
                
                total_weight = sum(m.get('weight', 1.0) for m in merged_metrics.values())
                weighted_sum = sum(m.get('score', 0) * m.get('weight', 1.0) for m in merged_metrics.values())
                new_total_score = weighted_sum / total_weight if total_weight > 0 else 0
                
                merged_result = {
                    'total_score': new_total_score,
                    'evaluation_time': datetime.now().isoformat(),
                    'metrics': merged_metrics
                }
                
                all_results[case_id] = merged_result
                
                metrics_dict = {}
                for name, m in merged_metrics.items():
                    if name == 'information_recall':
                        details = m.get('details', {})
                        components = details.get('components', {})
                        
                        lc = components.get('long_context', {})
                        if lc.get('available'):
                            metrics_dict['information_recall_longcontext'] = lc.get('score', 0)
                        
                        src = components.get('source_documents', {})
                        if src.get('available'):
                            metrics_dict['information_recall_source'] = src.get('score', 0)
                        
                        metrics_dict['information_recall'] = m.get('score', 0)
                    else:
                        metrics_dict[name] = m.get('score', 0)
                
                summary_scores.append({
                    'case_id': case_id,
                    'total_score': new_total_score,
                    'metrics': metrics_dict
                })
                
                logger.info(f"  ✅ Merged result. New total score: {new_total_score:.1f}")
            else:
                report = runner.generate_report(result)
                logger.info(report)
                
                all_results[case_id] = result.to_dict()
                
                metrics_dict = {}
                for name, r in result.results.items():
                    if name == 'information_recall':
                        details = r.details
                        components = details.get('components', {})
                        
                        lc = components.get('long_context', {})
                        if lc.get('available'):
                            metrics_dict['information_recall_longcontext'] = lc.get('score', 0)
                        
                        src = components.get('source_documents', {})
                        if src.get('available'):
                            metrics_dict['information_recall_source'] = src.get('score', 0)
                        
                        metrics_dict['information_recall'] = r.score
                    else:
                        metrics_dict[name] = r.score
                
                summary_scores.append({
                    'case_id': case_id,
                    'total_score': result.total_score,
                    'metrics': metrics_dict
                })
                
                if save_to_result_dir:
                    
                    eval_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    
                    for name, r in result.results.items():
                        metric_path = case_dir / f"eval_{name}_{eval_timestamp}.json"
                        metric_path.write_text(r.to_json(), encoding='utf-8')
                    
                    (case_dir / "eval_report.txt").write_text(
                        report, encoding='utf-8'
                    )
                    
                    logger.info(f"  ✅ Evaluation result saved to result dir: {case_dir}")
                    logger.info(f"     Saved {len(result.results)} individual metric files")
                
                if output_dir:
                    case_output_dir = output_dir / case_id
                    case_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    (case_output_dir / "combined_result.json").write_text(
                        result.to_json(), encoding='utf-8'
                    )
                    
                    (case_output_dir / "evaluation_report.txt").write_text(
                        report, encoding='utf-8'
                    )
                
        except Exception as e:
            logger.error(f"  ❌ Error evaluating case {case_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 EVALUATION SUMMARY")
    logger.info("=" * 60)
    
    if summary_scores:
        avg_total = sum(s['total_score'] for s in summary_scores) / len(summary_scores)
        
        logger.info(f"\n📈 Average Total Score: {avg_total:.1f}/100")
        if skipped_existing > 0:
            logger.info(f"   (Including {skipped_existing} previously evaluated cases)")
        logger.info(f"\n📋 Individual Scores:")
        
        for s in summary_scores:
            logger.info(f"  Case {s['case_id']}: {s['total_score']:.1f}/100")
            for metric, score in s['metrics'].items():
                logger.info(f"    - {metric}: {score:.1f}")
        
        logger.info(f"\n📊 Average Scores by Dimension:")
        metric_names = list(summary_scores[0]['metrics'].keys())
        
        anomaly_cases = {}  # {metric: [case_ids]}
        
        for metric in metric_names:
            valid_scores = [s['metrics'].get(metric, 0) for s in summary_scores 
                           if s['metrics'].get(metric, 0) >= 0]
            
            anomaly_case_ids = [s['case_id'] for s in summary_scores 
                               if s['metrics'].get(metric, 0) < 0]
            if anomaly_case_ids:
                anomaly_cases[metric] = anomaly_case_ids
            
            if valid_scores:
                avg = sum(valid_scores) / len(valid_scores)
                logger.info(f"  - {metric}: {avg:.1f}/100 (valid: {len(valid_scores)}/{len(summary_scores)})")
            else:
                logger.info(f"  - {metric}: N/A (no valid scores)")
        
        if anomaly_cases:
            logger.warning("")
            logger.warning("⚠️  ANOMALY DETECTION")
            logger.warning("=" * 60)
            logger.warning("The following cases have anomalous scores (< 0) and need to be re-evaluated:")
            logger.warning("")
            for metric, case_ids in anomaly_cases.items():
                logger.warning(f"  {metric}:")
                for case_id in case_ids:
                    logger.warning(f"    - Case {case_id}")
            logger.warning("")
            logger.warning("Please re-run evaluation for these cases to get valid scores.")
            logger.warning("=" * 60)
        
    else:
        logger.warning("  ⚠️ No cases were successfully evaluated")
    
    return all_results

def discover_models_and_contexts(result_base: Path) -> List[tuple]:
    """
    
    Returns:
        List of (context_size, model) tuples
    """
    combinations = []
    
    if not result_base.exists():
        logger.error(f"Error: Result base directory not found: {result_base}")
        return combinations
    
    for context_dir in sorted(result_base.iterdir()):
        if not context_dir.is_dir():
            continue
        
        context_size = context_dir.name
        
        for model_dir in sorted(context_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            
            model = model_dir.name
            
            case_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            if case_dirs:
                combinations.append((context_size, model))
    
    return combinations

def evaluate_single_config(
    config: Tuple[str, str],
    result_base: Path,
    insights_dir: Path,
    datasets_dir: Path,
    checklist_dir: Path,
    output_dir: Path,
    skip_existing: bool,
    metrics: list = None
) -> Tuple[str, str, Dict]:
    """
    
    Returns:
        (context_size, model, result_dict)
    """
    context_size, model = config
    result_dir = result_base / context_size / model
    eval_output_dir = output_dir / context_size / model
    
    case_dirs = [d for d in result_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"[{context_size}/{model}] Starting evaluation...")
    logger.info(f"[{context_size}/{model}] Cases found: {len(case_dirs)}")
    if metrics:
        logger.info(f"[{context_size}/{model}] Metrics: {', '.join(metrics)}")
    logger.info("=" * 60)
    
    try:
        run_batch_evaluation(
            result_dir=result_dir,
            insights_dir=insights_dir,
            output_dir=eval_output_dir,
            checklist_dir=checklist_dir,
            datasets_dir=datasets_dir,
            context_size=context_size,
            skip_existing=skip_existing,
            metrics=metrics
        )
        
        summary_path = eval_output_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
                
            result = {
                'total_cases': summary_data.get('total_cases', 0),
                'average_total_score': summary_data.get('average_total_score', 0),
                'average_by_dimension': summary_data.get('average_by_dimension', {})
            }
            logger.info(f"✅ [{context_size}/{model}] Completed! Score: {result['average_total_score']:.1f}")
            return context_size, model, result
        
        logger.warning(f"⚠️ [{context_size}/{model}] No summary.json found")
        return context_size, model, None
        
    except Exception as e:
        logger.error(f"❌ [{context_size}/{model}] Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return context_size, model, None

def run_all_evaluations(
    result_base: Path,
    insights_dir: Path,
    datasets_dir: Path = None,
    checklist_dir: Path = None,
    output_dir: Path = None,
    skip_existing: bool = True,
    max_workers: int = 4,
    metrics: list = None,
    exclude_models: list = None,
    include_models: list = None,
    include_cases: list = None,
    include_context_sizes: list = None
) -> Dict:
    """
    
    Args:
    
    Returns:
        Dict with all results
    """
    if output_dir is None:
        output_dir = Path("evaluation_logs") / result_base.name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging()
    
    combinations = discover_models_and_contexts(result_base)
    
    if not combinations:
        logger.error("No model/context combinations found!")
        return {}
    
    if include_context_sizes:
        original_count = len(combinations)
        combinations = [(cs, m) for cs, m in combinations if cs in include_context_sizes]
        included_count = len(combinations)
        logger.info(f"✅ Including only {included_count} configurations matching context sizes: {', '.join(include_context_sizes)}")
    
    if include_models:
        original_count = len(combinations)
        combinations = [(cs, m) for cs, m in combinations if m in include_models]
        included_count = len(combinations)
        logger.info(f"✅ Including only {included_count} configurations matching models: {', '.join(include_models)}")
    elif exclude_models:
        original_count = len(combinations)
        combinations = [(cs, m) for cs, m in combinations if m not in exclude_models]
        excluded_count = original_count - len(combinations)
        if excluded_count > 0:
            logger.info(f"⏭️ Excluded {excluded_count} configurations matching models: {', '.join(exclude_models)}")
    
    logger.info("=" * 60)
    logger.info("DISCOVERED CONFIGURATIONS")
    logger.info("=" * 60)
    logger.info(f"Result base: {result_base}")
    logger.info(f"Found {len(combinations)} configurations:")
    for context_size, model in combinations:
        logger.info(f"  - {context_size}/{model}")
    logger.info(f"Max workers: {max_workers}")
    if metrics:
        logger.info(f"Metrics: {', '.join(metrics)}")
    else:
        logger.info(f"Metrics: all ({', '.join(ALL_METRICS)})")
    if exclude_models:
        logger.info(f"Excluded models: {', '.join(exclude_models)}")
    if include_cases:
        logger.info(f"Include cases: {', '.join(include_cases)}")
    logger.info("=" * 60)
    logger.info("")
    
    total_cases = 0
    for context_size, model in combinations:
        result_dir = result_base / context_size / model
        case_dirs = [d for d in result_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if include_cases:
            case_dirs = [d for d in case_dirs if d.name in include_cases]
        total_cases += len(case_dirs)
    
    logger.info(f"📊 Total cases to process: {total_cases}")
    logger.info("")
    
    global_progress.init(total_configs=len(combinations), total_cases=total_cases)
    
    current_run_results = {}  # {(context_size, model): {case_id: {metric: score}}}
    merged_results = {}  # {(context_size, model): {case_id: {metric: score}}}
    
    all_results = {}
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    evaluate_single_config_v2,
                    config,
                    result_base,
                    insights_dir,
                    datasets_dir,
                    checklist_dir,
                    output_dir,
                    skip_existing,
                    metrics,
                    include_cases
                ): config
                for config in combinations
            }
            
            for future in concurrent.futures.as_completed(futures):
                config = futures[future]
                try:
                    context_size, model, current_scores, merged_scores = future.result()
                    
                    global_progress.update_config(increment=1)
                    
                    if current_scores:
                        current_run_results[(context_size, model)] = current_scores
                    if merged_scores:
                        merged_results[(context_size, model)] = merged_scores
                        if context_size not in all_results:
                            all_results[context_size] = {}
                        all_results[context_size][model] = {
                            'total_cases': len(merged_scores),
                            'average_by_dimension': {}
                        }
                except Exception as e:
                    logger.error(f"❌ Error processing {config}: {e}")
                    global_progress.update_config(increment=1)
    finally:
        global_progress.close()
    
    datasets_name = datasets_dir.name if datasets_dir else output_dir.name
    models_evaluated = list(set(m for _, m in combinations))
    generate_incremental_summary_report(
        current_run_results=current_run_results,
        merged_results=merged_results,
        result_base=result_base,
        datasets_name=datasets_name,
        metrics_run=metrics or ALL_METRICS,
        models_evaluated=models_evaluated
    )
    
    return all_results

def evaluate_single_config_v2(
    config: Tuple[str, str],
    result_base: Path,
    insights_dir: Path,
    datasets_dir: Path,
    checklist_dir: Path,
    output_dir: Path,
    skip_existing: bool,
    metrics: list = None,
    include_cases: list = None
) -> Tuple[str, str, Dict, Dict]:
    """
    
    Args:
    
    Returns:
        (context_size, model, current_run_scores, merged_scores)
    """
    context_size, model = config
    result_dir = result_base / context_size / model
    eval_output_dir = output_dir / context_size / model
    
    case_dirs = [d for d in result_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    
    if include_cases:
        case_dirs = [d for d in case_dirs if d.name in include_cases]
        logger.info(f"[{context_size}/{model}] Filtered to {len(case_dirs)} cases: {', '.join(d.name for d in case_dirs)}")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"[{context_size}/{model}] Starting evaluation...")
    logger.info(f"[{context_size}/{model}] Cases to evaluate: {len(case_dirs)}")
    if metrics:
        logger.info(f"[{context_size}/{model}] Metrics: {', '.join(metrics)}")
    logger.info("=" * 60)
    
    current_run_scores = {}  # {case_id: {metric: score}}
    merged_scores = {}  # {case_id: {metric: score}}
    
    try:
        incremental_mode = metrics is not None and len(metrics) < len(ALL_METRICS)
        
        from evaluators.utils.run_all import EvaluationRunner
        runner = EvaluationRunner(metrics=metrics)
        
        sorted_case_dirs = sorted(case_dirs, key=lambda x: x.name)
        
        config_name = f"{context_size}/{model}"
        global_progress.update_config(config_name=config_name)
        
        for case_dir in sorted_case_dirs:
            case_id = case_dir.name
            
            global_progress.update_case(case_id=f"{config_name}/{case_id}")
            
            final_report = case_dir / "final_report.md"
            if not final_report.exists():
                logger.warning(f"  ⚠️ Skipping case {case_id}: final_report.md not found")
                continue
            
            existing_metrics = set()
            existing_metric_data = {}
            requested_metrics = set(metrics) if metrics else set(ALL_METRICS)
            
            for metric_name in ALL_METRICS:
                metric_files = list(case_dir.glob(f"eval_{metric_name}_*.json"))
                if not metric_files:
                    metric_files = list(case_dir.glob(f"eval_{metric_name}.json"))
                
                if metric_files:
                    metric_files.sort(key=lambda x: x.name, reverse=True)
                    metric_file = metric_files[0]
                    try:
                        metric_data = json.loads(metric_file.read_text(encoding='utf-8'))
                        existing_metrics.add(metric_name)
                        existing_metric_data[metric_name] = metric_data
                    except Exception as e:
                        logger.warning(f"  ⚠️ Failed to load {metric_file.name} for case {case_id}: {e}")
            
            if skip_existing:
                failed_metrics = set()
                
                for metric_name in existing_metric_data.keys():
                    metric_data = existing_metric_data[metric_name]
                    details = metric_data.get('details', {})
                    
                    metric_json_str = json.dumps(metric_data, ensure_ascii=False).lower()
                    
                    error_patterns = []
                    
                    if 'parse error' in metric_json_str:
                        error_patterns.append('parse error')
                    
                    if 'api error' in metric_json_str:
                        error_patterns.append('api error')
                    
                    if '"explanation": "not verified"' in metric_json_str or '"explanation":"not verified"' in metric_json_str:
                        error_patterns.append('Not verified')
                    
                    score = metric_data.get('score', 0)
                    if score < 0:
                        error_patterns.append(f'negative score ({score})')
                    
                    if metric_name == 'information_recall':
                        details = metric_data.get('details', {})
                        long_context_score = details.get('long_context_score')
                        source_documents_score = details.get('source_documents_score')
                        insights_case_dir = insights_dir / case_id
                        
                        if long_context_score is None:
                            gold_lc_path = insights_case_dir / "gold_insights_from_long_context.json"
                            if gold_lc_path.exists():
                                try:
                                    gold_lc_data = json.loads(gold_lc_path.read_text(encoding='utf-8'))
                                    gold_insights = gold_lc_data.get('gold_insights', [])
                                    if gold_insights:
                                        error_patterns.append(f'long_context_score is null but insights now available ({len(gold_insights)} insights)')
                                except Exception:
                                    pass
                        
                        if source_documents_score is None:
                            gold_src_path = insights_case_dir / "gold_insights_from_source.json"
                            if gold_src_path.exists():
                                try:
                                    gold_src_data = json.loads(gold_src_path.read_text(encoding='utf-8'))
                                    gold_insights = gold_src_data.get('gold_insights', [])
                                    if gold_insights:
                                        error_patterns.append(f'source_documents_score is null but insights now available ({len(gold_insights)} insights)')
                                except Exception:
                                    pass
                    
                    if error_patterns:
                        if metric_name in requested_metrics:
                            failed_metrics.add(metric_name)
                            logger.info(f"  ⚠️ Case {case_id}: {metric_name} has errors ({', '.join(error_patterns)}), will re-run")
                        else:
                            logger.info(f"  ℹ️ Case {case_id}: {metric_name} has errors ({', '.join(error_patterns)}), but not in requested metrics")
                
                for metric_name in requested_metrics:
                    if metric_name in existing_metric_data:
                        metric_data = existing_metric_data[metric_name]
                        score = metric_data.get('score', 0)
                        if score < 0:
                            failed_metrics.add(metric_name)
                            logger.info(f"  ⚠️ Case {case_id}: {metric_name} has error score ({score}), will re-run")
                
                if requested_metrics.issubset(existing_metrics) and not failed_metrics:
                    logger.info(f"  ⏭️ Skipping case {case_id}: all requested metrics already evaluated ({len(existing_metrics)} metrics found)")
                    merged_scores[case_id] = {}
                    for name, m in existing_metric_data.items():
                        merged_scores[case_id][name] = m.get('score', 0)
                    continue
                else:
                    missing_metrics = requested_metrics - existing_metrics
                    metrics_to_run = missing_metrics | failed_metrics
                    
                    if failed_metrics:
                        logger.info(f"  🔄 Case {case_id}: will re-run failed metrics {failed_metrics}")
                    if missing_metrics:
                        logger.info(f"  📝 Case {case_id}: will evaluate missing metrics {missing_metrics}")
                    
                    from evaluators.utils.run_all import EvaluationRunner
                    runner = EvaluationRunner(metrics=list(metrics_to_run))
            
            insights_case_dir = insights_dir / case_id
            gold_path = insights_case_dir / "gold_insights_from_long_context.json" if insights_case_dir.exists() else None
            gold_source_path = insights_case_dir / "gold_insights_from_source.json" if insights_case_dir.exists() else None
            execution_log = case_dir / "execution_log.json"
            rag_queries = case_dir / "rag_queries.txt"
            
            long_context_path = None
            source_folder = None
            if datasets_dir:
                datasets_case_dir = datasets_dir / case_id
                if datasets_case_dir.exists():
                    long_context_file = datasets_case_dir / f"long_context_sampled_{context_size}.json"
                    if long_context_file.exists():
                        long_context_path = long_context_file
                    source_folder = datasets_case_dir
            
            checklist_path = None
            if checklist_dir:
                checklist_case_dir = checklist_dir / case_id
                if checklist_case_dir.exists():
                    checklist_path = checklist_case_dir / "checklist.json"
                    if not checklist_path.exists():
                        checklist_path = None
            
            query_file = None
            if datasets_dir:
                query_file = datasets_dir / "query.jsonl"
                if not query_file.exists():
                    query_file = None
            
            try:
                result = runner.run(
                    result_path=final_report,
                    gold_path=gold_path if gold_path and gold_path.exists() else None,
                    gold_source_path=gold_source_path if gold_source_path and gold_source_path.exists() else None,
                    execution_log_path=execution_log if execution_log.exists() else None,
                    checklist_path=checklist_path,
                    long_context_path=long_context_path,
                    source_folder=source_folder,
                    query_file=query_file,
                    case_id=case_id,
                    context_size=context_size,
                    rag_queries_path=rag_queries if rag_queries.exists() else None
                )
                
                current_run_scores[case_id] = {}
                for name, r in result.results.items():
                    current_run_scores[case_id][name] = r.score
                
                if incremental_mode and existing_metric_data:
                    merged_metrics = existing_metric_data.copy()
                    for name, r in result.results.items():
                        merged_metrics[name] = r.to_dict()
                    
                    total_weight = sum(m.get('weight', 1.0) for m in merged_metrics.values())
                    weighted_sum = sum(m.get('score', 0) * m.get('weight', 1.0) for m in merged_metrics.values())
                    new_total_score = weighted_sum / total_weight if total_weight > 0 else 0
                    
                    merged_result = {
                        'total_score': new_total_score,
                        'evaluation_time': datetime.now().isoformat(),
                        'metrics': merged_metrics
                    }
                    
                    eval_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    
                    for name, r in result.results.items():
                        metric_path = case_dir / f"eval_{name}_{eval_timestamp}.json"
                        metric_path.write_text(r.to_json(), encoding='utf-8')
                    
                    merged_scores[case_id] = {}
                    for name, m in merged_metrics.items():
                        merged_scores[case_id][name] = m.get('score', 0)
                    
                    logger.info(f"  ✅ Case {case_id}: merged result saved. Score: {new_total_score:.1f} (saved {len(result.results)} new metrics)")
                else:
                    eval_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    
                    for name, r in result.results.items():
                        metric_path = case_dir / f"eval_{name}_{eval_timestamp}.json"
                        metric_path.write_text(r.to_json(), encoding='utf-8')
                    
                    merged_scores[case_id] = {}
                    for name, r in result.results.items():
                        merged_scores[case_id][name] = r.score
                    
                    logger.info(f"  ✅ Case {case_id}: result saved. Score: {result.total_score:.1f}")
                
                global_progress.update_case(increment=1)
                    
            except Exception as e:
                logger.error(f"  ❌ Error evaluating case {case_id}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        logger.info(f"✅ [{context_size}/{model}] Completed!")
        return context_size, model, current_run_scores, merged_scores
        
    except Exception as e:
        logger.error(f"❌ [{context_size}/{model}] Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return context_size, model, {}, {}

def generate_incremental_summary_report(
    current_run_results: Dict,
    merged_results: Dict,
    result_base: Path,
    datasets_name: str,
    metrics_run: List[str],
    models_evaluated: List[str]
):
    """Generate an incremental summary report of evaluation results."""
    context_sizes = sorted(set(cs for cs, _ in merged_results.keys()), key=lambda x: int(x.replace('k', '')))
    
    md_lines = []
    md_lines.append('# Evaluation Run Summary')
    md_lines.append('')
    md_lines.append(f'**Dataset:** `{datasets_name}`')
    md_lines.append(f'**Result Base:** `{result_base}`')
    md_lines.append(f'**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    md_lines.append(f'**Metrics Run:** `{", ".join(metrics_run)}`')
    md_lines.append(f'**Models Evaluated:** `{", ".join(models_evaluated)}`')
    md_lines.append('')
    md_lines.append('---')
    md_lines.append('')
    
    def get_information_recall_subscores(result_base: Path, context_size: str, model: str, case_id: str):
        case_dir = result_base / context_size / model / case_id
        if not case_dir.exists():
            return None, None
        
        eval_files = list(case_dir.glob("eval_information_recall*.json"))
        if not eval_files:
            return None, None
        
        eval_files.sort(key=lambda x: x.name, reverse=True)
        eval_file = eval_files[0]
        
        try:
            with open(eval_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            details = data.get('details', {})
            long_context_score = details.get('long_context_score')
            source_documents_score = details.get('source_documents_score')
            return long_context_score, source_documents_score
        except Exception:
            return None, None
    
    md_lines.append('## 📊 This Run Results')
    md_lines.append('')
    md_lines.append(f'> Metrics evaluated in this run: **{", ".join(metrics_run)}**')
    md_lines.append('')
    
    for metric in metrics_run:
        if metric == 'information_recall':
            md_lines.append(f'### information_recall (Long Context)')
            md_lines.append('')
            
            header = '| Model |'
            separator = '|-------|'
            for cs in context_sizes:
                header += f' {cs} |'
                separator += '------|'
            md_lines.append(header)
            md_lines.append(separator)
            
            for model in models_evaluated:
                row = f'| {model} |'
                for cs in context_sizes:
                    key = (cs, model)
                    if key in current_run_results:
                        scores = []
                        for case_id in current_run_results[key].keys():
                            lc_score, _ = get_information_recall_subscores(result_base, cs, model, case_id)
                            if lc_score is not None:
                                scores.append(lc_score)
                        if scores:
                            avg = sum(scores) / len(scores)
                            row += f' {avg:.1f} ({len(scores)}) |'
                        else:
                            row += ' N/A |'
                    else:
                        row += ' N/A |'
                md_lines.append(row)
            md_lines.append('')
            
            md_lines.append(f'### information_recall (Source Documents)')
            md_lines.append('')
            
            header = '| Model |'
            separator = '|-------|'
            for cs in context_sizes:
                header += f' {cs} |'
                separator += '------|'
            md_lines.append(header)
            md_lines.append(separator)
            
            for model in models_evaluated:
                row = f'| {model} |'
                for cs in context_sizes:
                    key = (cs, model)
                    if key in current_run_results:
                        scores = []
                        for case_id in current_run_results[key].keys():
                            _, src_score = get_information_recall_subscores(result_base, cs, model, case_id)
                            if src_score is not None:
                                scores.append(src_score)
                        if scores:
                            avg = sum(scores) / len(scores)
                            row += f' {avg:.1f} ({len(scores)}) |'
                        else:
                            row += ' N/A |'
                    else:
                        row += ' N/A |'
                md_lines.append(row)
            md_lines.append('')
        else:
            md_lines.append(f'### {metric}')
            md_lines.append('')
            
            header = '| Model |'
            separator = '|-------|'
            for cs in context_sizes:
                header += f' {cs} |'
                separator += '------|'
            md_lines.append(header)
            md_lines.append(separator)
            
            for model in models_evaluated:
                row = f'| {model} |'
                for cs in context_sizes:
                    key = (cs, model)
                    if key in current_run_results:
                        scores = [s.get(metric, 0) for s in current_run_results[key].values() if metric in s]
                        if scores:
                            avg = sum(scores) / len(scores)
                            row += f' {avg:.1f} ({len(scores)}) |'
                        else:
                            row += ' N/A |'
                    else:
                        row += ' N/A |'
                md_lines.append(row)
            md_lines.append('')
    
    md_lines.append('---')
    md_lines.append('')
    md_lines.append('## 📈 Complete Merged Results')
    md_lines.append('')
    md_lines.append('> All metrics after merging with previous evaluations')
    md_lines.append('')
    
    all_metrics = set()
    for scores_dict in merged_results.values():
        for case_scores in scores_dict.values():
            all_metrics.update(case_scores.keys())
    all_metrics = sorted(list(all_metrics))
    
    md_lines.append('### 📊 Overall Score (Average of All Metrics)')
    md_lines.append('')
    md_lines.append('> Total score is the average of all available metrics for each model')
    md_lines.append('')
    
    header = '| Model |'
    separator = '|-------|'
    for cs in context_sizes:
        header += f' {cs} |'
        separator += '------|'
    md_lines.append(header)
    md_lines.append(separator)
    
    for model in models_evaluated:
        row = f'| {model} |'
        for cs in context_sizes:
            key = (cs, model)
            if key in merged_results:
                case_avg_scores = []
                for case_id, case_scores in merged_results[key].items():
                    valid_scores = [s for s in case_scores.values() if s >= 0]
                    if valid_scores:
                        case_avg = sum(valid_scores) / len(valid_scores)
                        case_avg_scores.append(case_avg)
                
                if case_avg_scores:
                    overall_avg = sum(case_avg_scores) / len(case_avg_scores)
                    row += f' {overall_avg:.1f} ({len(case_avg_scores)}) |'
                else:
                    row += ' N/A |'
            else:
                row += ' N/A |'
        md_lines.append(row)
    md_lines.append('')
    
    for metric in all_metrics:
        if metric == 'information_recall':
            md_lines.append(f'### information_recall (Long Context)')
            md_lines.append('')
            
            header = '| Model |'
            separator = '|-------|'
            for cs in context_sizes:
                header += f' {cs} |'
                separator += '------|'
            md_lines.append(header)
            md_lines.append(separator)
            
            for model in models_evaluated:
                row = f'| {model} |'
                for cs in context_sizes:
                    key = (cs, model)
                    if key in merged_results:
                        scores = []
                        for case_id in merged_results[key].keys():
                            lc_score, _ = get_information_recall_subscores(result_base, cs, model, case_id)
                            if lc_score is not None:
                                scores.append(lc_score)
                        if scores:
                            avg = sum(scores) / len(scores)
                            row += f' {avg:.1f} ({len(scores)}) |'
                        else:
                            row += ' N/A |'
                    else:
                        row += ' N/A |'
                md_lines.append(row)
            md_lines.append('')
            
            md_lines.append(f'### information_recall (Source Documents)')
            md_lines.append('')
            
            header = '| Model |'
            separator = '|-------|'
            for cs in context_sizes:
                header += f' {cs} |'
                separator += '------|'
            md_lines.append(header)
            md_lines.append(separator)
            
            for model in models_evaluated:
                row = f'| {model} |'
                for cs in context_sizes:
                    key = (cs, model)
                    if key in merged_results:
                        scores = []
                        for case_id in merged_results[key].keys():
                            _, src_score = get_information_recall_subscores(result_base, cs, model, case_id)
                            if src_score is not None:
                                scores.append(src_score)
                        if scores:
                            avg = sum(scores) / len(scores)
                            row += f' {avg:.1f} ({len(scores)}) |'
                        else:
                            row += ' N/A |'
                    else:
                        row += ' N/A |'
                md_lines.append(row)
            md_lines.append('')
        else:
            md_lines.append(f'### {metric}')
            md_lines.append('')
            
            header = '| Model |'
            separator = '|-------|'
            for cs in context_sizes:
                header += f' {cs} |'
                separator += '------|'
            md_lines.append(header)
            md_lines.append(separator)
            
            for model in models_evaluated:
                row = f'| {model} |'
                for cs in context_sizes:
                    key = (cs, model)
                    if key in merged_results:
                        scores = [s.get(metric, 0) for s in merged_results[key].values() if metric in s]
                        if scores:
                            avg = sum(scores) / len(scores)
                            row += f' {avg:.1f} ({len(scores)}) |'
                        else:
                            row += ' N/A |'
                    else:
                        row += ' N/A |'
                md_lines.append(row)
            md_lines.append('')
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info('\n'.join(md_lines))

def generate_summary_report(all_results: Dict, output_dir: Path, datasets_name: str = None):
    """Generate a summary report of all evaluation results."""
    
    if not all_results:
        logger.warning("No results to summarize")
        return
    
    context_sizes = sorted(all_results.keys(), key=lambda x: int(x.replace('k', '')))
    models = set()
    for cs in context_sizes:
        models.update(all_results[cs].keys())
    models = sorted(list(models))
    
    all_dimensions = set()
    for cs in context_sizes:
        for model in models:
            if model in all_results.get(cs, {}):
                all_dimensions.update(all_results[cs][model].get('average_by_dimension', {}).keys())
    all_dimensions = sorted(list(all_dimensions))
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if datasets_name is None:
        datasets_name = output_dir.name
    
    md_lines = []
    md_lines.append('# Evaluation Summary Report')
    md_lines.append('')
    md_lines.append(f'**Dataset:** `{datasets_name}`')
    md_lines.append(f'**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    md_lines.append(f'**Output Directory:** `{output_dir}`')
    md_lines.append('')
    md_lines.append('---')
    md_lines.append('')
    
    md_lines.append('## 📊 Average Total Score')
    md_lines.append('')
    header = '| Model |'
    separator = '|-------|'
    for cs in context_sizes:
        header += f' {cs} |'
        separator += '------|'
    md_lines.append(header)
    md_lines.append(separator)
    
    for model in models:
        row = f'| {model} |'
        for cs in context_sizes:
            if cs in all_results and model in all_results[cs]:
                score = all_results[cs][model].get('average_total_score', 0)
                row += f' {score:.1f} |'
            else:
                row += ' N/A |'
        md_lines.append(row)
    md_lines.append('')
    
    md_lines.append('## 📊 Information Recall (Long Context)')
    md_lines.append('')
    header = '| Model |'
    separator = '|-------|'
    for cs in context_sizes:
        header += f' {cs} |'
        separator += '------|'
    md_lines.append(header)
    md_lines.append(separator)
    
    for model in models:
        row = f'| {model} |'
        for cs in context_sizes:
            if cs in all_results and model in all_results[cs]:
                score = all_results[cs][model].get('average_by_dimension', {}).get('information_recall_longcontext', None)
                if score is not None:
                    row += f' {score:.1f} |'
                else:
                    scores = []
                    eval_dir = output_dir / cs / model
                    if eval_dir.exists():
                        for case_dir in eval_dir.iterdir():
                            if case_dir.is_dir() and case_dir.name.isdigit():
                                combined_path = case_dir / 'combined_result.json'
                                if combined_path.exists():
                                    try:
                                        with open(combined_path, 'r', encoding='utf-8') as f:
                                            combined = json.load(f)
                                        ir = combined.get('metrics', {}).get('information_recall', {})
                                        details = ir.get('details', {})
                                        components = details.get('components', {})
                                        lc = components.get('long_context', {})
                                        if lc.get('available'):
                                            scores.append(lc.get('score', 0))
                                    except Exception:
                                        pass
                    if scores:
                        row += f' {sum(scores)/len(scores):.1f} |'
                    else:
                        row += ' N/A |'
            else:
                row += ' N/A |'
        md_lines.append(row)
    md_lines.append('')
    
    md_lines.append('## 📊 Information Recall (Source Documents)')
    md_lines.append('')
    header = '| Model |'
    separator = '|-------|'
    for cs in context_sizes:
        header += f' {cs} |'
        separator += '------|'
    md_lines.append(header)
    md_lines.append(separator)
    
    for model in models:
        row = f'| {model} |'
        for cs in context_sizes:
            if cs in all_results and model in all_results[cs]:
                score = all_results[cs][model].get('average_by_dimension', {}).get('information_recall_source', None)
                if score is not None:
                    row += f' {score:.1f} |'
                else:
                    scores = []
                    eval_dir = output_dir / cs / model
                    if eval_dir.exists():
                        for case_dir in eval_dir.iterdir():
                            if case_dir.is_dir() and case_dir.name.isdigit():
                                combined_path = case_dir / 'combined_result.json'
                                if combined_path.exists():
                                    try:
                                        with open(combined_path, 'r', encoding='utf-8') as f:
                                            combined = json.load(f)
                                        ir = combined.get('metrics', {}).get('information_recall', {})
                                        details = ir.get('details', {})
                                        components = details.get('components', {})
                                        src = components.get('source_documents', {})
                                        if src.get('available'):
                                            scores.append(src.get('score', 0))
                                    except Exception:
                                        pass
                    if scores:
                        row += f' {sum(scores)/len(scores):.1f} |'
                    else:
                        row += ' N/A |'
            else:
                row += ' N/A |'
        md_lines.append(row)
    md_lines.append('')
    
    excluded_dimensions = {'information_recall_longcontext', 'information_recall_source'}
    for dimension in all_dimensions:
        if dimension in excluded_dimensions:
            continue
        md_lines.append(f'## 📊 {dimension.replace("_", " ").title()}')
        md_lines.append('')
        
        header = '| Model |'
        separator = '|-------|'
        for cs in context_sizes:
            header += f' {cs} |'
            separator += '------|'
        md_lines.append(header)
        md_lines.append(separator)
        
        for model in models:
            row = f'| {model} |'
            for cs in context_sizes:
                if cs in all_results and model in all_results[cs]:
                    score = all_results[cs][model].get('average_by_dimension', {}).get(dimension, 0)
                    row += f' {score:.1f} |'
                else:
                    row += ' N/A |'
            md_lines.append(row)
        md_lines.append('')
    
    md_lines.append('## 📊 Total Cases Evaluated')
    md_lines.append('')
    header = '| Model |'
    separator = '|-------|'
    for cs in context_sizes:
        header += f' {cs} |'
        separator += '------|'
    md_lines.append(header)
    md_lines.append(separator)
    
    for model in models:
        row = f'| {model} |'
        for cs in context_sizes:
            if cs in all_results and model in all_results[cs]:
                cases = all_results[cs][model].get('total_cases', 0)
                row += f' {cases} |'
            else:
                row += ' N/A |'
        md_lines.append(row)
    md_lines.append('')
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info('\n'.join(md_lines))

def main():
    parser = argparse.ArgumentParser(description="Batch run evaluators")
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    single_parser = subparsers.add_parser('single', help='Evaluate a single result directory')
    single_parser.add_argument("--result-dir", type=str, required=True, 
                        help="Path to result directory containing case folders")
    single_parser.add_argument("--insights-dir", type=str, required=True,
                        help="Path to insights directory containing gold insights")
    single_parser.add_argument("--datasets-dir", type=str,
                        help="Path to datasets directory containing long_context and useful_search files")
    single_parser.add_argument("--context-size", type=str, default="32k",
                        choices=["32k", "64k", "128k", "256k", "512k"],
                        help="Context size for long_context file (default: 32k)")
    single_parser.add_argument("--checklist-dir", type=str,
                        help="Path to checklist directory containing checklist.json files")
    single_parser.add_argument("--output-dir", type=str,
                        help="Output directory for evaluation results")
    single_parser.add_argument("--no-skip", action="store_true",
                        help="Do not skip already evaluated cases (re-evaluate all)")
    single_parser.add_argument("--metrics", "-m", type=str, nargs='+',
                        choices=ALL_METRICS,
                        help=f"Specific metrics to run (default: all). Available: {', '.join(ALL_METRICS)}")
    
    all_parser = subparsers.add_parser('all', help='Evaluate all models and context sizes')
    all_parser.add_argument("--result-base", "-r", type=str, required=True,
                        help="Base directory containing context_size/model subdirectories")
    all_parser.add_argument("--insights-dir", "-i", type=str, default=None,
                        help="Path to insights directory (default: auto-detect from datasets-dir, e.g., ground_truth/<datasets_name>)")
    all_parser.add_argument("--datasets-dir", "-d", type=str, required=True,
                        help="Path to datasets directory (e.g., datasets_batch4)")
    all_parser.add_argument("--checklist-dir", "-c", type=str, default=None,
                        help="Path to checklist directory (default: auto-detect, e.g., ground_truth/<datasets_name>)")
    all_parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Output directory for evaluation logs (default: evaluation_logs/<result-base-name>)")
    all_parser.add_argument("--no-skip", action="store_true",
                        help="Do not skip already evaluated cases (re-evaluate all)")
    all_parser.add_argument("--workers", "-w", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    all_parser.add_argument("--metrics", "-m", type=str, nargs='+',
                        choices=ALL_METRICS,
                        help=f"Specific metrics to run (default: all). Available: {', '.join(ALL_METRICS)}")
    all_parser.add_argument("--exclude-models", "-e", type=str, nargs='+',
                        help="Models to exclude from evaluation (e.g., claude37_sonnet)")
    all_parser.add_argument("--include-models", type=str, nargs='+',
                        help="Only include these models (e.g., gpt-4.1 Qwen3-30B-A3B)")
    all_parser.add_argument("--include-cases", type=str, nargs='+',
                        help="Only include these case IDs (e.g., 024 012 013)")
    all_parser.add_argument("--include-context-sizes", type=str, nargs='+',
                        help="Only include these context sizes (e.g., 64k 512k)")
    
    two_phase_parser = subparsers.add_parser('two-phase', help='Two-phase evaluation: fast metrics first, then factual_accuracy with rate limiting')
    two_phase_parser.add_argument("--result-base", "-r", type=str, required=True,
                        help="Base directory containing context_size/model subdirectories")
    two_phase_parser.add_argument("--insights-dir", "-i", type=str, default="ground_truth/datasets_batch2",
                        help="Path to insights directory (default: ground_truth/datasets_batch2)")
    two_phase_parser.add_argument("--datasets-dir", "-d", type=str, default="datasets_batch2",
                        help="Path to datasets directory (default: datasets_batch2)")
    two_phase_parser.add_argument("--checklist-dir", "-c", type=str, default="ground_truth/datasets_batch2",
                        help="Path to checklist directory (default: ground_truth/datasets_batch2)")
    two_phase_parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Output directory for evaluation logs (default: evaluation_logs/<result-base-name>)")
    two_phase_parser.add_argument("--no-skip", action="store_true",
                        help="Do not skip already evaluated cases (re-evaluate all)")
    two_phase_parser.add_argument("--phase1-workers", type=int, default=15,
                        help="Number of parallel workers for phase 1 (non-Gemini metrics, default: 15)")
    two_phase_parser.add_argument("--phase2-workers", type=int, default=1,
                        help="Number of parallel workers for phase 2 (factual_accuracy, default: 1)")
    two_phase_parser.add_argument("--gemini-rpm", type=int, default=10,
                        help="Gemini API rate limit (requests per minute, default: 10)")
    two_phase_parser.add_argument("--exclude-models", "-e", type=str, nargs='+',
                        help="Models to exclude from evaluation (e.g., claude37_sonnet)")
    two_phase_parser.add_argument("--skip-phase1", action="store_true",
                        help="Skip phase 1 (non-Gemini metrics)")
    two_phase_parser.add_argument("--skip-phase2", action="store_true",
                        help="Skip phase 2 (factual_accuracy)")
    
    args = parser.parse_args()
    
    if args.command == 'single':
        result_dir = Path(args.result_dir)
        insights_dir = Path(args.insights_dir)
        datasets_dir = Path(args.datasets_dir) if args.datasets_dir else None
        checklist_dir = Path(args.checklist_dir) if args.checklist_dir else None
        output_dir = Path(args.output_dir) if args.output_dir else result_dir / "eval_results"
        
        setup_logging(output_dir)
        
        run_batch_evaluation(
            result_dir=result_dir, 
            insights_dir=insights_dir, 
            output_dir=output_dir, 
            checklist_dir=checklist_dir,
            datasets_dir=datasets_dir,
            context_size=args.context_size,
            skip_existing=not args.no_skip,
            metrics=args.metrics
        )
    
    elif args.command == 'all':
        result_base = Path(args.result_base)
        datasets_dir = Path(args.datasets_dir)
        
        datasets_name = datasets_dir.name
        
        if args.insights_dir:
            insights_dir = Path(args.insights_dir)
        else:
            insights_dir = Path("evaluation/insights") / datasets_name
            logger.info(f"📁 Auto-detected insights_dir: {insights_dir}")
        
        if args.checklist_dir:
            checklist_dir = Path(args.checklist_dir)
        else:
            checklist_dir = Path("evaluation/checklists") / datasets_name
            logger.info(f"📁 Auto-detected checklist_dir: {checklist_dir}")
        
        output_dir = Path(args.output_dir) if args.output_dir else None
        
        run_all_evaluations(
            result_base=result_base,
            insights_dir=insights_dir,
            datasets_dir=datasets_dir,
            checklist_dir=checklist_dir,
            output_dir=output_dir,
            skip_existing=not args.no_skip,
            max_workers=args.workers,
            metrics=args.metrics,
            exclude_models=args.exclude_models,
            include_models=args.include_models,
            include_cases=args.include_cases,
            include_context_sizes=args.include_context_sizes
        )
    
    elif args.command == 'two-phase':
        import os
        
        result_base = Path(args.result_base)
        insights_dir = Path(args.insights_dir)
        datasets_dir = Path(args.datasets_dir) if args.datasets_dir else None
        checklist_dir = Path(args.checklist_dir) if args.checklist_dir else None
        output_dir = Path(args.output_dir) if args.output_dir else Path("evaluation_logs") / result_base.name
        
        output_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(output_dir)
        
        logger.info("=" * 60)
        logger.info("🚀 TWO-PHASE EVALUATION")
        logger.info("=" * 60)
        logger.info(f"Result base: {result_base}")
        logger.info(f"Output dir: {output_dir}")
        logger.info(f"Phase 1 workers: {args.phase1_workers}")
        logger.info(f"Phase 2 workers: {args.phase2_workers}")
        logger.info(f"Gemini RPM: {args.gemini_rpm}")
        logger.info("=" * 60)
        
        non_gemini_metrics = ['information_recall', 'overall_quality', 'format_compliance', 'citation_coverage', 'tool_usage']
        
        if not args.skip_phase1:
            logger.info("")
            logger.info("=" * 60)
            logger.info("📊 PHASE 1: Non-Gemini Metrics (High Concurrency)")
            logger.info(f"   Metrics: {', '.join(non_gemini_metrics)}")
            logger.info(f"   Workers: {args.phase1_workers}")
            logger.info("=" * 60)
            
            run_all_evaluations(
                result_base=result_base,
                insights_dir=insights_dir,
                datasets_dir=datasets_dir,
                checklist_dir=checklist_dir,
                output_dir=output_dir,
                skip_existing=not args.no_skip,
                max_workers=args.phase1_workers,
                metrics=non_gemini_metrics,
                exclude_models=args.exclude_models
            )
            
            logger.info("")
            logger.info("✅ Phase 1 completed!")
        else:
            logger.info("⏭️ Skipping Phase 1")
        
        if not args.skip_phase2:
            logger.info("")
            logger.info("=" * 60)
            logger.info("📊 PHASE 2: Factual Accuracy (Rate Limited)")
            logger.info(f"   Metrics: factual_accuracy")
            logger.info(f"   Workers: {args.phase2_workers}")
            logger.info(f"   Gemini RPM: {args.gemini_rpm}")
            logger.info("=" * 60)
            
            run_all_evaluations(
                result_base=result_base,
                insights_dir=insights_dir,
                datasets_dir=datasets_dir,
                checklist_dir=checklist_dir,
                output_dir=output_dir,
                skip_existing=not args.no_skip,
                max_workers=args.phase2_workers,
                metrics=['factual_accuracy'],
                exclude_models=args.exclude_models
            )
            
            logger.info("")
            logger.info("✅ Phase 2 completed!")
        else:
            logger.info("⏭️ Skipping Phase 2")
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("🎉 TWO-PHASE EVALUATION COMPLETED!")
        logger.info(f"📁 Results saved to: {output_dir}")
        logger.info("=" * 60)
    
    else:
        parser.print_help()
        print("\n" + "=" * 60)
        print("Example usage:")
        print("=" * 60)
        print("\nSingle model evaluation:")
        print("uv run python eval.py single \\")
        print("    --result-dir result/20251227-new/datasets_batch2/32k/gpt-4.1 \\")
        print("    --insights-dir insights/batch2_insights")
        print("\nAll models evaluation:")
        print("uv run python eval.py all \\")
        print("    --result-base result/20251227-new/datasets_batch2 \\")
        print("    --output-dir evaluation_logs/batch_20251227_new")

if __name__ == "__main__":
    main()
