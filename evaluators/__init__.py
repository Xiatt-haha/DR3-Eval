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

"""Evaluator module - each evaluator handles one metric."""

from .utils.base import EvalConfig, EvalResult, BaseEvaluator, EvaluationAPIError
from .utils.llm_client import LLMClient
from .utils.document_loader import DocumentLoader
from .information_recall import InformationRecallEvaluator
from .factual_accuracy import FactualAccuracyAgentEvaluator
from .depth_quality import OverallQualityEvaluator
from .format_compliance import FormatComplianceEvaluator
from .citation_coverage import CitationCoverageEvaluator
from .utils.run_all import EvaluationRunner, CombinedEvalResult

__all__ = [
    'EvalConfig',
    'EvalResult',
    'BaseEvaluator',
    'EvaluationAPIError',
    'LLMClient',
    'DocumentLoader',
    'InformationRecallEvaluator',
    'FactualAccuracyAgentEvaluator',
    'OverallQualityEvaluator',
    'FormatComplianceEvaluator',
    'CitationCoverageEvaluator',
    'EvaluationRunner',
    'CombinedEvalResult',
]
