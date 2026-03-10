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
    python -m evaluators.information_recall \
        --result examples/001/final_report.md \
        --gold examples/001/gold_insights.json \
        --output examples/001/eval_information_recall.json
        
    python -m evaluators.information_recall \
        --result result/001/final_report.md \
        --data-dir datasets_batch4 \
        --task 001 \
"""

import logging
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .utils.base import BaseEvaluator, EvalConfig, EvalResult, EvaluationAPIError
from .utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

DEFAULT_INSIGHTS_DIR = "evaluation/insights"

class InformationRecallEvaluator(BaseEvaluator):
    """Information recall evaluator using ternary scoring (1.0 / 0.5 / 0.0)."""
    
    metric_name = "information_recall"
    weight = 1.0
    BATCH_SIZE = 10
    
    def __init__(self, config: Optional[EvalConfig] = None):
        super().__init__(config)
        self.llm = LLMClient(self.config)
    
    def evaluate(self, result_text: str, 
                 gold_insights: List[Dict] = None,
                 source_gold_insights: List[Dict] = None,
                 query: str = None,
                 data_dir: str = None,
                 task_number: str = None,
                 insights_base_dir: str = None,
                 auto_extract: bool = True,
                 evaluate_only: str = None,
                 existing_source_score: float = None,
                 existing_source_details: Dict = None,
                 **kwargs) -> EvalResult:
        gold_insights = gold_insights or []
        source_gold_insights = source_gold_insights or []
        self.query = query or ""
        
        if not gold_insights and not source_gold_insights:
            if data_dir and task_number:
                print(f"  No insights provided, attempting to load or generate...")
                gold_insights, source_gold_insights = self._load_or_generate_insights(
                    data_dir=data_dir,
                    task_number=task_number,
                    query=query,
                    insights_base_dir=insights_base_dir,
                    auto_extract=auto_extract
                )
        
        if not gold_insights and not source_gold_insights:
            error_msg = "No gold insights provided (neither from long_context nor source_documents)"
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
        
        if evaluate_only == 'long_context':
            long_score, long_details, has_long = self._evaluate_component(
                result_text, gold_insights, 'long_context'
            )
            if existing_source_score is not None:
                source_score = existing_source_score
                source_details = existing_source_details or {}
                has_source = True
                print(f"  Using existing source_documents_score: {source_score:.1f}")
            else:
                source_score, source_details, has_source = 100.0, {}, False
        elif evaluate_only == 'source_documents':
            source_score, source_details, has_source = self._evaluate_component(
                result_text, source_gold_insights, 'source_documents'
            )
            long_score, long_details, has_long = 100.0, {}, False
        else:
            long_score, long_details, has_long = self._evaluate_component(
                result_text, gold_insights, 'long_context'
            )
            source_score, source_details, has_source = self._evaluate_component(
                result_text, source_gold_insights, 'source_documents'
            )
        
        details = {
            'long_context_score': long_score if has_long else None,
            'source_documents_score': source_score if has_source else None,
            'components': {
                'long_context': {
                    'score': long_score,
                    'details': long_details,
                    'available': has_long
                },
                'source_documents': {
                    'score': source_score,
                    'details': source_details,
                    'available': has_source
                }
            },
            'note': 'Scores are reported separately for long_context and source_documents'
        }
        
        if has_long and has_source:
            combined_score = (long_score + source_score) / 2
        elif has_long:
            combined_score = long_score
        elif has_source:
            combined_score = source_score
        else:
            combined_score = 100.0
        
        return EvalResult(
            metric_name=self.metric_name,
            score=combined_score,
            details=details,
            weight=self.weight
        )
    
    def _load_or_generate_insights(
        self,
        data_dir: str,
        task_number: str,
        query: str = None,
        insights_base_dir: str = None,
        auto_extract: bool = True
    ) -> Tuple[List[Dict], List[Dict]]:
        if insights_base_dir is None:
            insights_base_dir = DEFAULT_INSIGHTS_DIR
        
        dataset_name = os.path.basename(os.path.normpath(data_dir))
        insights_dir = os.path.join(insights_base_dir, dataset_name, task_number)
        
        long_context_file = os.path.join(insights_dir, "gold_insights_from_long_context.json")
        source_file = os.path.join(insights_dir, "gold_insights_from_source.json")
        
        long_context_insights = []
        source_insights = []
        
        long_context_exists = os.path.exists(long_context_file)
        source_exists = os.path.exists(source_file)
        
        if long_context_exists:
            print(f"  Loading existing long-context insights from: {long_context_file}")
            with open(long_context_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                long_context_insights = data.get("gold_insights", [])
            print(f"  Loaded {len(long_context_insights)} long-context insights")
        
        if source_exists:
            print(f"  Loading existing source insights from: {source_file}")
            with open(source_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                source_insights = data.get("gold_insights", [])
            print(f"  Loaded {len(source_insights)} source insights")
        
        if long_context_exists and source_exists:
            return long_context_insights, source_insights
        
        if auto_extract and (not long_context_exists or not source_exists):
            print(f"  Some insights missing, attempting to extract...")
            
            if not query:
                query = self._load_query_from_file(data_dir, task_number)
            
            if query:
                try:
                    from .utils.extract_insights_combined import extract_all_insights
                    
                    output_dir = os.path.join(insights_base_dir, dataset_name)
                    extracted_long, extracted_source = extract_all_insights(
                        data_dir=data_dir,
                        output_dir=output_dir,
                        task_number=task_number,
                        query=query,
                        mode="all"
                    )
                    
                    if not long_context_exists and extracted_long:
                        long_context_insights = extracted_long
                    if not source_exists and extracted_source:
                        source_insights = extracted_source
                        
                except ImportError as e:
                    print(f"  Warning: Could not import extraction utilities: {e}")
                except Exception as e:
                    print(f"  Warning: Failed to extract insights: {e}")
            else:
                print(f"  Warning: No query provided, cannot extract insights")
        
        return long_context_insights, source_insights
    
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
    
    def _evaluate_component(self, result_text: str, 
                           gold_insights: List[Dict], 
                           label: str) -> Tuple[float, Dict, bool]:
        if not gold_insights:
            return 100.0, {
                'message': f'No gold insights provided for {label}',
                'evaluation_result': {
                    'covered_insights': [],
                    'total_insights': 0,
                    'fully_covered_count': 0,
                    'partially_covered_count': 0,
                    'not_covered_count': 0,
                    'recall_percentage': 100.0
                }
            }, False
        
        covered_insights = []
        total = len(gold_insights)
        
        print(f"  Evaluating {total} insights for {label} (batch size: {self.BATCH_SIZE})...")
        
        for batch_start in range(0, total, self.BATCH_SIZE):
            batch_end = min(batch_start + self.BATCH_SIZE, total)
            batch = gold_insights[batch_start:batch_end]
            batch_num = batch_start // self.BATCH_SIZE + 1
            total_batches = (total + self.BATCH_SIZE - 1) // self.BATCH_SIZE
            
            print(f"    Batch {batch_num}/{total_batches}: insights {batch_start+1}-{batch_end}")
            
            batch_results = self._verify_batch_insights(result_text, batch, batch_start)
            covered_insights.extend(batch_results)
        
        fully_covered_count = sum(1 for item in covered_insights if item.get('score') == 1.0)
        not_covered_count = sum(1 for item in covered_insights if item.get('score') != 1.0)
        
        recall_percentage = 100.0 * fully_covered_count / total if total > 0 else 100.0
        
        evaluation_result = {
            'covered_insights': covered_insights,
            'total_insights': total,
            'fully_covered_count': fully_covered_count,
            'not_covered_count': not_covered_count,
            'recall_percentage': recall_percentage
        }
        
        scoring_note = 'Binary scoring: Only 1.0 (fully covered) counts as covered. Partial coverage (0.5) and not covered (0.0) both count as 0. Recall = fully_covered / total * 100'
        
        details = {
            'evaluation_result': evaluation_result,
            'recall_rate': recall_percentage / 100,
            'fully_covered_count': fully_covered_count,
            'not_covered_count': not_covered_count,
            'total_count': total,
            'original_gold_count': len(gold_insights),
            'source_label': label,
            'scoring_note': scoring_note
        }
        
        return recall_percentage, details, True
    
    def _verify_batch_insights(self, result_text: str, batch: List[Dict], start_index: int) -> List[Dict]:
        insights_list = "\n".join([
            f"{i+1}. {insight['insight']}"
            for i, insight in enumerate(batch)
        ])
        
        prompt = f"""You are an intelligent assistant that labels atomic insights based on whether they can be extracted from a given report.

## Search Query/Task
{self.query or "No specific query provided"}

## Insights to Label (Total: {len(batch)})
{insights_list}

## Report/Passage
{result_text}

TASK: Label each insight as 1.0 (support), 0.5 (partial_support), or 0.0 (not_support) based on whether the insight can be extracted from the report.

TERNARY SCORING SYSTEM (三级评分系统)

【SCORE 1.0 - FULLY COVERED (完全覆盖)】
Definition: The report EXPLICITLY and COMPLETELY covers the insight's core meaning.

Requirements for 1.0 (ALL must be met):
✅ The core information point is clearly stated in the report
✅ No significant details are missing
✅ The meaning is unambiguous and complete
✅ A reader would understand the full insight from the report alone

Examples of 1.0:
- Insight: "上海2023年垃圾分类覆盖率达到95%"
  Report: "上海市2023年生活垃圾分类覆盖率已达95%以上" → 1.0 (exact match)
- Insight: "EPNP算法用于相机位姿估计"
  Report: "使用EPNP算法进行相机姿态估计" → 1.0 (same meaning, different wording)

【SCORE 0.5 - PARTIALLY COVERED (部分覆盖)】
Definition: The report contains RELATED content but is INCOMPLETE or VAGUE.

Conditions for 0.5 (any of these):
⚠️ Only part of the core meaning is covered (missing key details)
⚠️ The topic is mentioned but specifics are absent
⚠️ Related concept exists but not the exact point
⚠️ Generalization without the specific insight
⚠️ The connection requires inference (not explicit)

Examples of 0.5:
- Insight: "上海2023年垃圾分类覆盖率达到95%"
  Report: "上海垃圾分类取得显著成效" → 0.5 (topic covered, but no specific percentage)
- Insight: "德国采用双轨制回收系统"
  Report: "发达国家有成熟的回收体系" → 0.5 (generalization, no specific country/system)
- Insight: "AI伦理需要建立问责机制"
  Report: "AI发展需要考虑伦理问题" → 0.5 (related topic, but no specific mechanism)

【SCORE 0.0 - NOT COVERED (未覆盖)】
Definition: The report does NOT contain the insight's information.

Conditions for 0.0:
❌ The topic is completely absent from the report
❌ Only unrelated keywords appear
❌ The information contradicts the insight
❌ No semantic connection exists

Examples of 0.0:
- Insight: "新加坡的垃圾焚烧发电技术"
  Report: (never mentions Singapore) → 0.0
- Insight: "666训练计划的具体内容"
  Report: (never mentions "666" or this training plan) → 0.0

THREE-STEP JUDGMENT PROCESS (三步判断流程)

For EACH insight, follow these steps:

**STEP 1: Extract Core Meaning (提取核心含义)**
- Identify the KEY information points (usually 1-3)
- Note any specific details (numbers, names, dates, methods)
- Example: "上海2023年垃圾分类覆盖率达到95%" 
  → Core: [上海, 2023年, 垃圾分类, 覆盖率, 95%]

**STEP 2: Search Report for Coverage (搜索报告中的覆盖情况)**
- Look for exact matches first
- Then look for semantic equivalents
- Note what IS found and what is MISSING

**STEP 3: Apply Scoring Rules (应用评分规则)**
- If ALL core points are explicitly covered → 1.0
- If SOME core points are covered OR topic is mentioned without specifics → 0.5
- If NO core points are found → 0.0

SEMANTIC EQUIVALENCE GUIDELINES (语义等价判断标准)

✅ The following are considered SEMANTICALLY EQUIVALENT (score 1.0):

a) Synonym Substitution (同义替换):
   - "法律" ≈ "法规" ≈ "条例" ≈ "规定"
   - "成效显著" ≈ "效果明显" ≈ "取得成功"
   - "技术多样化" ≈ "多种技术" ≈ "技术种类丰富"

b) Hypernym/Hyponym (上下位概念):
   - "EPNP算法" → "PnP算法" (EPNP is a type of PnP) → 0.5 (too general)
   - "德国、日本" → "发达国家" / "国际经验" → 0.5 (generalization)
   - "高德地图" → "导航应用" (if context is clear) → 0.5 (less specific)

c) Paraphrasing (改写和概括):
   - "建立AI伦理审查委员会" ≈ "设立AI伦理委员会进行审查" → 1.0
   - "六个伦理设计细节" ≈ "六大伦理问题" → 1.0 (if all 6 are covered)

d) Information Fusion (信息融合):
   - Insight content is integrated into a larger discussion
   - Core information points are preserved with different expression

❌ The following are NOT semantically equivalent (score 0.0 or 0.5):

a) Completely Different Topics:
   - Insight discusses topic A, report discusses topic B
   - No semantic overlap whatsoever

b) Keyword Overlap Only:
   - Same words appear but with completely different meanings
   - Example: Insight says "Apple company", report says "apple fruit"

c) Missing Information:
   - Core information points from insight are completely absent in report
   - Example: Insight mentions specific number "666", report never mentions it

d) Contradictory Information:
   - Report information contradicts the insight
   - Example: Insight says "successful", report says "failed"

BOUNDARY CASE HANDLING (边界情况处理规则)

When uncertain, follow this priority:

1. If >50% of core meaning is covered → 1.0
2. If reasonable semantic connection exists → 1.0
3. If only weak connection or keyword overlap → 0.5
4. If no connection at all → 0.0

Principle: Prefer false positives over false negatives
(The goal of recall assessment is to check if information is missing)

DECISION BOUNDARY EXAMPLES (边界情况示例)

| Insight | Report Content | Score | Reason |
|---------|---------------|-------|--------|
| "上海95%覆盖率" | "上海95%覆盖率" | 1.0 | Exact match |
| "上海95%覆盖率" | "上海覆盖率很高" | 0.5 | Missing specific number |
| "上海95%覆盖率" | "中国垃圾分类进展" | 0.5 | Related topic, no Shanghai detail |
| "上海95%覆盖率" | (not mentioned) | 0.0 | Absent |
| "EPNP算法" | "PnP算法" | 0.5 | Related but not specific variant |
| "EPNP算法" | "EPNP算法" | 1.0 | Exact match |
| "德国双轨制" | "国际经验" | 0.5 | Generalization |
| "德国双轨制" | "德国的双轨制回收" | 1.0 | Specific match |
| "法律规定" | "法规要求" | 1.0 | Synonym substitution |
| "建立委员会" | "设立委员会" | 1.0 | Paraphrasing |

RESPONSE FORMAT

Respond ONLY with valid JSON (no markdown, no extra text):
{{
    "results": [
        {{
            "id": 1,
            "core_points": ["point1", "point2"],
            "found_in_report": "[quote or describe what was found]",
            "missing_points": ["what is missing, if any"],
            "score": 1.0,
            "reasoning": "[brief reason in Chinese, <30 words]"
        }},
        {{
            "id": 2,
            "core_points": ["point1", "point2"],
            "found_in_report": "Related content but incomplete",
            "missing_points": ["specific detail X"],
            "score": 0.5,
            "reasoning": "[brief reason in Chinese, <30 words]"
        }},
        {{
            "id": 3,
            "core_points": ["point1"],
            "found_in_report": "NOT FOUND",
            "missing_points": ["all points missing"],
            "score": 0.0,
            "reasoning": "[brief reason in Chinese, <30 words]"
        }}
    ]
}}

You MUST provide exactly {len(batch)} results. Use ONLY scores 1.0, 0.5, or 0.0."""
        
        system_prompt = """You are an intelligent assistant that labels atomic insights based on whether they can be extracted from a given report.

CORE PRINCIPLE: TERNARY SCORING - Use 1.0, 0.5, or 0.0
- Score 1.0 (support): The insight CAN BE FULLY EXTRACTED from the report - all core information is explicitly present
- Score 0.5 (partial_support): The insight CAN BE PARTIALLY EXTRACTED - some information is present but incomplete
- Score 0.0 (not_support): The insight CANNOT BE EXTRACTED - the information is absent or unrelated

SCORING GUIDELINES:

【1.0 - FULLY COVERED】
✅ All core information points are clearly stated
✅ No significant details are missing
✅ The meaning is unambiguous and complete
✅ A reader would understand the full insight from the report alone

Examples:
- "上海2023年垃圾分类覆盖率达到95%" → Report: "上海市2023年生活垃圾分类覆盖率已达95%以上" → 1.0
- "EPNP算法用于相机位姿估计" → Report: "使用EPNP算法进行相机姿态估计" → 1.0

【0.5 - PARTIALLY COVERED】
⚠️ Only part of the core meaning is covered (missing key details)
⚠️ The topic is mentioned but specifics are absent
⚠️ Related concept exists but not the exact point
⚠️ Generalization without the specific insight

Examples:
- "上海2023年垃圾分类覆盖率达到95%" → Report: "上海垃圾分类取得显著成效" → 0.5 (no specific percentage)
- "德国采用双轨制回收系统" → Report: "发达国家有成熟的回收体系" → 0.5 (generalization)
- "AI伦理需要建立问责机制" → Report: "AI发展需要考虑伦理问题" → 0.5 (related but no specific mechanism)

【0.0 - NOT COVERED】
❌ The topic is completely absent from the report
❌ Only unrelated keywords appear
❌ The information contradicts the insight
❌ No semantic connection exists

KEY PRINCIPLE: Be precise with scoring. Use 1.0 only when the insight is fully and explicitly covered. Use 0.5 for partial or vague coverage. Use 0.0 when absent."""
        
        result = self.llm.call(
            system=system_prompt,
            user=prompt
        )
        
        if not result or 'results' not in result:
            print(f"      ⚠️ Detailed format parse failed, trying compact format...")
            result = self._verify_batch_insights_compact(result_text, batch, start_index)
        
        if not result or 'results' not in result:
            error_msg = f"LLM verification failed for batch (insights {start_index+1}-{start_index+len(batch)})"
            print(f"      ❌ ERROR: {error_msg}")
            raise EvaluationAPIError(
                error_msg,
                metric_name="information_recall",
                details={'batch_start': start_index, 'batch_size': len(batch)}
            )
        
        batch_results = []
        llm_results = result.get('results', [])
        
        for i, insight in enumerate(batch):
            insight_id = start_index + i + 1
            insight_text = insight['insight']
            
            llm_result = None
            for r in llm_results:
                if r.get('id') == i + 1:
                    llm_result = r
                    break
            
            if llm_result:
                score = llm_result.get('score', 0.0)
                if score not in [0.0, 0.5, 1.0]:
                    if score >= 0.75:
                        score = 1.0
                    elif score >= 0.25:
                        score = 0.5
                    else:
                        score = 0.0
                
                core_meaning = llm_result.get('core_meaning', '')
                found_in_report = llm_result.get('found_in_report', '')
                
                batch_results.append({
                    'id': insight_id,
                    'score': score,
                    'covered': score == 1.0,
                    'core_meaning': core_meaning,
                    'found_in_report': found_in_report,
                    'explanation': llm_result.get('explanation', ''),
                    'insight_text': insight_text[:100] + '...' if len(insight_text) > 100 else insight_text
                })
            else:
                batch_results.append({
                    'id': insight_id,
                    'score': 0.0,
                    'covered': False,
                    'core_meaning': '',
                    'found_in_report': 'NOT FOUND',
                    'explanation': 'No result from LLM for this insight',
                    'insight_text': insight_text[:100] + '...' if len(insight_text) > 100 else insight_text
                })
        
        return batch_results
    
    def _verify_batch_insights_compact(self, result_text: str, batch: List[Dict], start_index: int) -> Optional[Dict]:
        insights_list = "\n".join([
            f"{i+1}. {insight['insight']}"
            for i, insight in enumerate(batch)
        ])
        
        compact_prompt = f"""Label each insight as 1.0 (fully covered), 0.5 (partially covered), or 0.0 (not covered) based on the report.

## Insights ({len(batch)} total)
{insights_list}

## Report
{result_text}

SCORING:
- 1.0: All core information explicitly present in report
- 0.5: Related content exists but incomplete/vague
- 0.0: Information absent from report

OUTPUT FORMAT (JSON only, no explanation):
{{"results": [{{"id": 1, "score": 1.0}}, {{"id": 2, "score": 0.5}}, ...]}}

Provide exactly {len(batch)} results."""

        compact_system = """You are a precise evaluator. Output ONLY valid JSON with scores (1.0, 0.5, or 0.0) for each insight. No explanations needed."""
        
        for attempt in range(3):
            result = self.llm.call(
                system=compact_system,
                user=compact_prompt
            )
            if result and 'results' in result:
                print(f"      ✓ Compact format succeeded (attempt {attempt + 1})")
                return result
            print(f"      Retrying compact format {attempt + 1}/3...")
        
        return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate information recall")
    parser.add_argument("--result", type=str, required=True, help="Path to result markdown file")
    parser.add_argument("--gold", type=str, help="Path to gold insights JSON (from long context)")
    parser.add_argument("--gold-source", type=str, help="Path to source gold insights JSON")
    parser.add_argument("--insights-dir", type=str, help="Path to insights directory containing gold_insights_from_longcontext.json and gold_insights_from_source.json")
    parser.add_argument("--data-dir", type=str, help="Path to data directory (for auto-extraction)")
    parser.add_argument("--task", type=str, help="Task number (for auto-extraction)")
    parser.add_argument("--query", type=str, help="Query text (for auto-extraction)")
    parser.add_argument("--output", type=str, help="Output file for evaluation result")
    
    args = parser.parse_args()
    
    result_text = Path(args.result).read_text(encoding='utf-8')
    
    gold_insights = []
    source_gold_insights = []
    
    if args.insights_dir:
        insights_dir = Path(args.insights_dir)
        longcontext_file = insights_dir / "gold_insights_from_long_context.json"
        source_file = insights_dir / "gold_insights_from_source.json"
        
        if longcontext_file.exists():
            gold_data = json.loads(longcontext_file.read_text(encoding='utf-8'))
            gold_insights = gold_data.get('gold_insights', [])
        if source_file.exists():
            source_data = json.loads(source_file.read_text(encoding='utf-8'))
            source_gold_insights = source_data.get('gold_insights', [])
    else:
        if args.gold:
            gold_data = json.loads(Path(args.gold).read_text(encoding='utf-8'))
            gold_insights = gold_data.get('gold_insights', [])
        if args.gold_source:
            source_data = json.loads(Path(args.gold_source).read_text(encoding='utf-8'))
            source_gold_insights = source_data.get('gold_insights', [])
    
    evaluator = InformationRecallEvaluator()
    result = evaluator.evaluate(
        result_text,
        gold_insights=gold_insights,
        source_gold_insights=source_gold_insights,
        query=args.query,
        data_dir=args.data_dir,
        task_number=args.task
    )
    
    print(f"\n📋 Information Recall Score: {result.score:.1f}/100")
    print(result.to_json())
    
    if args.output:
        evaluator.save_result(result, Path(args.output))
        print(f"\n📄 Result saved to: {args.output}")

if __name__ == "__main__":
    main()
