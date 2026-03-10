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

"""Citation coverage evaluator."""

import json
import re
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import glob

from .utils.base import BaseEvaluator, EvalConfig, EvalResult
from .utils.document_loader import normalize_title, fuzzy_title_match


logger = logging.getLogger(__name__)


class CitationCoverageEvaluator(BaseEvaluator):
    """Evaluates whether the report cites all required documents."""
    
    metric_name = "citation_coverage"
    weight = 1.0
    
    USER_DOC_EXTENSIONS = {'.md', '.txt', '.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls'}
    
    def evaluate(self, result_text: str,
                 useful_search_path: Optional[Path] = None,
                 dataset_dir: Optional[Path] = None,
                 context_size: Optional[str] = None,
                 required_titles: Optional[List[str]] = None,
                 **kwargs) -> EvalResult:
        titles = required_titles or []
        title_sources = {}
        
        if dataset_dir and dataset_dir.exists():
            useful_search_file = dataset_dir / "useful_search.jsonl"
            if not useful_search_file.exists():
                useful_search_file = dataset_dir / "useful_search.json"
            
            if useful_search_file.exists():
                useful_titles = self._load_useful_search(useful_search_file)
                for title in useful_titles:
                    if title not in titles:
                        titles.append(title)
                        title_sources[title] = "useful_search"
                logger.info(f"Loaded {len(useful_titles)} required titles from {useful_search_file.name}")
                
                user_doc_titles = self._load_user_document_titles(dataset_dir)
                for title in user_doc_titles:
                    if title not in titles:
                        titles.append(title)
                        title_sources[title] = "user_document"
            else:
                error_msg = f"useful_search.jsonl not found in {dataset_dir}"
                logger.error(error_msg)
                return EvalResult(
                    metric_name=self.metric_name,
                    score=-1,
                    details={'error': error_msg, 'status': 'failed', 'dataset_dir': str(dataset_dir)},
                    weight=self.weight
                )
        elif useful_search_path and useful_search_path.exists():
            titles = self._load_useful_search(useful_search_path)
            for title in titles:
                title_sources[title] = "useful_search"
        
        if not titles:
            error_msg = "No required titles specified (dataset_dir not found or empty)"
            logger.error(error_msg)
            return EvalResult(
                metric_name=self.metric_name,
                score=-1,
                details={'error': error_msg, 'status': 'failed'},
                weight=self.weight
            )
        
        if "No final answer" in result_text:
            return EvalResult(
                metric_name=self.metric_name,
                score=0.0,
                details={
                    'message': 'Empty report (No final answer)',
                    'cited': [], 'missing': titles,
                    'total_required': len(titles), 'total_cited': 0
                },
                weight=self.weight
            )
        
        result = self._check_citations_with_extraction(result_text, titles)
        
        cited_count = len(result['cited'])
        total_count = len(titles)
        coverage_rate = (cited_count / total_count * 100) if total_count > 0 else 100.0
        
        useful_search_titles = [t for t in titles if title_sources.get(t) == "useful_search"]
        user_doc_titles = [t for t in titles if title_sources.get(t) == "user_document"]
        
        us_cited = [t for t in result['cited'] if title_sources.get(t) == "useful_search"]
        us_missing = [t for t in result['missing'] if title_sources.get(t) == "useful_search"]
        us_total = len(useful_search_titles)
        us_score = (len(us_cited) / us_total * 100) if us_total > 0 else -1
        
        ud_cited = [t for t in result['cited'] if title_sources.get(t) == "user_document"]
        ud_missing = [t for t in result['missing'] if title_sources.get(t) == "user_document"]
        ud_total = len(user_doc_titles)
        ud_score = (len(ud_cited) / ud_total * 100) if ud_total > 0 else -1
        
        details = {
            'cited': result['cited'],
            'missing': result['missing'],
            'total_required': total_count,
            'total_cited': cited_count,
            'coverage_rate': coverage_rate,
            'full_coverage': cited_count == total_count,
            'extracted_citations': result.get('extracted_citations', []),
            'match_details': result.get('match_details', {}),
            'score_breakdown': {
                'useful_search': {
                    'score': us_score, 'cited': us_cited, 'missing': us_missing,
                    'cited_count': len(us_cited), 'total_count': us_total
                },
                'user_document': {
                    'score': ud_score, 'cited': ud_cited, 'missing': ud_missing,
                    'cited_count': len(ud_cited), 'total_count': ud_total
                }
            }
        }
        
        us_score_str = f"{us_score:.1f}" if us_score >= 0 else "N/A"
        ud_score_str = f"{ud_score:.1f}" if ud_score >= 0 else "N/A"
        logger.info(f"Citation Coverage: {coverage_rate:.1f}/100 ({cited_count}/{total_count})")
        logger.info(f"  - Useful Search (Long Context): {us_score_str}/100 ({len(us_cited)}/{us_total})")
        logger.info(f"  - User Documents: {ud_score_str}/100 ({len(ud_cited)}/{ud_total})")
        
        return EvalResult(
            metric_name=self.metric_name,
            score=coverage_rate,
            details=details,
            weight=self.weight
        )
    
    def _load_useful_search(self, path: Path) -> List[str]:
        """Load title list from useful_search.json or useful_search.jsonl."""
        try:
            content = path.read_text(encoding='utf-8')
            titles = []
            
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and item.get("title"):
                            titles.append(item["title"])
                    return titles
            except json.JSONDecodeError:
                pass
            
            for line in content.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if isinstance(item, list):
                        for obj in item:
                            if isinstance(obj, dict) and obj.get("title"):
                                titles.append(obj["title"])
                    elif isinstance(item, dict) and item.get("title"):
                        titles.append(item["title"])
                except json.JSONDecodeError:
                    continue
            
            return titles
        except Exception:
            return []
    
    def _load_long_context_titles(self, dataset_dir: Path, context_size: Optional[str] = None) -> List[str]:
        """Load document titles from long_context_sampled_*.json files."""
        titles = []
        seen = set()
        
        try:
            if context_size:
                pattern = f"long_context_sampled_{context_size}.json"
                files = list(dataset_dir.glob(pattern))
            else:
                files = list(dataset_dir.glob("long_context_sampled_*.json"))
            
            for file_path in files:
                try:
                    content = file_path.read_text(encoding='utf-8')
                    data = json.loads(content)
                    
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and item.get("title"):
                                title = item["title"]
                                if title not in seen:
                                    titles.append(title)
                                    seen.add(title)
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
                    continue
            
            logger.info(f"Loaded {len(titles)} titles from long_context files in {dataset_dir}")
        except Exception as e:
            logger.warning(f"Failed to load long_context titles from {dataset_dir}: {e}")
        
        return titles
    
    def _load_user_document_titles(self, dataset_dir: Path) -> List[str]:
        """Load user document filenames from the dataset directory."""
        titles = []
        
        try:
            for file_path in dataset_dir.iterdir():
                if file_path.is_file():
                    suffix = file_path.suffix.lower()
                    if suffix in self.USER_DOC_EXTENSIONS:
                        filename = file_path.name
                        if filename.startswith('long_context_sampled_'):
                            continue
                        if filename in ('useful_search.jsonl', 'useful_search.json', 'task.json', 'task.md'):
                            continue
                        titles.append(filename)
            
            logger.info(f"Loaded {len(titles)} user document titles from {dataset_dir}")
        except Exception as e:
            logger.warning(f"Failed to load user document titles from {dataset_dir}: {e}")
        
        return titles
    
    def _extract_explicit_citations(self, content: str) -> List[str]:
        """Extract all explicit citations from the report text.
        
        Supported formats:
        - [long_context: "..."]
        - [filename.md, page X]
        - [source: "..."]
        - [Image: ...]
        - [general title] (at least 4 chars)
        """
        citations = []
        seen = set()
        
        # [long_context: "..."] format
        patterns_long_context = [
            r'\[long_context:\s*["\']([^"\']+)["\']',
            r'\[long_context:\s*[\u201c]([^\u201d]+)[\u201d]',
        ]
        
        for pattern in patterns_long_context:
            matches = re.findall(pattern, content)
            for match in matches:
                title = re.sub(r',\s*chunk\s*\d+$', '', match).strip()
                if title and title not in seen:
                    citations.append(title)
                    seen.add(title)
        
        # [filename.md, page X] format
        pattern_md_page = r'\[([^\[\]]+\.(?:md|txt|pdf|docx?)),\s*\u7b2c[\d\-]+\u9875\]'
        matches = re.findall(pattern_md_page, content)
        for match in matches:
            title = match.strip()
            if title and title not in seen:
                citations.append(title)
                seen.add(title)
        
        # [source: "..."] format
        patterns_source = [
            r'\[source:\s*["\']([^"\']+)["\']',
            r'\[source:\s*[\u201c]([^\u201d]+)[\u201d]',
        ]
        
        for pattern in patterns_source:
            matches = re.findall(pattern, content)
            for match in matches:
                title = re.sub(r',\s*chunk\s*\d+$', '', match).strip()
                if title and title not in seen:
                    citations.append(title)
                    seen.add(title)
        
        # [Image: ...] format
        pattern_image = r'\[Image:\s*([^\[\]]+)\]'
        matches = re.findall(pattern_image, content)
        for match in matches:
            title = match.strip()
            if title and title not in seen:
                citations.append(title)
                seen.add(title)
        
        # General [title] format (at least 4 chars, excluding known non-citation patterns)
        pattern_general = r'\[([^\[\]]{4,})\]'
        matches = re.findall(pattern_general, content)
        for match in matches:
            title = match.strip()
            if title.startswith('long_context:') or title.startswith('source:') or title.startswith('Image:'):
                continue
            if re.match(r'^[\d\s]+$', title) or re.match(r'^[a-zA-Z]$', title):
                continue
            if '(' in title or ')' in title:
                continue
            skip_patterns = ['\u6ce8', '\u56fe', '\u8868', '\u9644\u5f55', 'TODO', 'NOTE', 'WARNING', 'TIP', 'IMPORTANT']
            if any(title.startswith(p) for p in skip_patterns):
                continue
            
            if title and title not in seen:
                citations.append(title)
                seen.add(title)
        
        return citations
    
    def _extract_core_identifiers(self, title: str) -> List[str]:
        """Extract core identifiers (e.g. paper abbreviations) from a title."""
        identifiers = []
        
        abbreviations = re.findall(r'[A-Z][A-Z0-9\-]+[A-Z0-9]', title)
        identifiers.extend(abbreviations)
        
        versioned = re.findall(r'\d+[A-Z\-]+[A-Z]+|[A-Z]+\-\d+[A-Z]+', title)
        identifiers.extend(versioned)
        
        parenthetical = re.findall(r'[\uff08(]([^\uff09)]+)[\uff09)]', title)
        identifiers.extend(parenthetical)
        
        unique_identifiers = []
        for ident in identifiers:
            if len(ident) >= 3 and ident not in unique_identifiers:
                unique_identifiers.append(ident)
        
        return unique_identifiers
    
    def _match_title_to_citation(self, required_title: str, citation: str) -> Tuple[bool, str]:
        """Check if a required title matches a citation using multi-layer matching."""
        required_norm = normalize_title(required_title)
        citation_norm = normalize_title(citation)
        
        main_title = required_title.split("-")[0].strip() if "-" in required_title else required_title
        main_title_norm = normalize_title(main_title)
        
        if required_norm == citation_norm or main_title_norm == citation_norm:
            return True, "exact"
        
        if required_norm in citation_norm or citation_norm in required_norm:
            return True, "contains"
        if main_title_norm in citation_norm or citation_norm in main_title_norm:
            return True, "contains_main"
        
        core_identifiers = self._extract_core_identifiers(required_title)
        for identifier in core_identifiers:
            identifier_lower = identifier.lower()
            if len(identifier_lower) >= 3 and identifier_lower in citation_norm.lower():
                return True, "core_identifier"
        
        if fuzzy_title_match(required_title, citation, threshold=0.6):
            return True, "fuzzy"
        
        return False, ""
    
    def _check_citations_with_extraction(self, result_text: str, required_titles: List[str]) -> Dict:
        """Check which required titles are cited in the report."""
        extracted_citations = self._extract_explicit_citations(result_text)
        
        cited = []
        missing = []
        match_details = {}
        
        result_text_norm = normalize_title(result_text)
        
        for title in required_titles:
            found = False
            match_type = ""
            matched_citation = ""
            
            # Check against extracted citations
            for citation in extracted_citations:
                is_match, m_type = self._match_title_to_citation(title, citation)
                if is_match:
                    found = True
                    match_type = m_type
                    matched_citation = citation
                    break
            
            # Fallback: check if title appears directly in report text
            if not found:
                title_norm = normalize_title(title)
                if title_norm and len(title_norm) >= 4 and title_norm in result_text_norm:
                    found = True
                    match_type = "text_contains"
                    matched_citation = title
            
            # Fallback: check core identifiers in full text
            if not found:
                core_ids = self._extract_core_identifiers(title)
                for cid in core_ids:
                    if len(cid) >= 4 and cid.lower() in result_text.lower():
                        found = True
                        match_type = "core_id_in_text"
                        matched_citation = cid
                        break
            
            if found:
                cited.append(title)
                match_details[title] = {
                    'match_type': match_type,
                    'matched_citation': matched_citation
                }
            else:
                missing.append(title)
                match_details[title] = {'match_type': 'not_found', 'matched_citation': ''}
        
        return {
            'cited': cited,
            'missing': missing,
            'extracted_citations': extracted_citations,
            'match_details': match_details
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate citation coverage")
    parser.add_argument("--result", type=str, required=True, help="Path to result markdown file")
    parser.add_argument("--dataset-dir", type=str, help="Path to dataset directory")
    parser.add_argument("--useful-search", type=str, help="Path to useful_search.json")
    parser.add_argument("--context-size", type=str, default=None, help="Context size (e.g. 32k)")
    parser.add_argument("--output", type=str, help="Output file for evaluation result")
    
    args = parser.parse_args()
    
    result_text = Path(args.result).read_text(encoding='utf-8')
    
    evaluator = CitationCoverageEvaluator()
    result = evaluator.evaluate(
        result_text,
        useful_search_path=Path(args.useful_search) if args.useful_search else None,
        dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
        context_size=args.context_size
    )
    
    print(f"\nCitation Coverage Score: {result.score:.1f}/100")
    print(result.to_json())
    
    if args.output:
        evaluator.save_result(result, Path(args.output))
        print(f"\nResult saved to: {args.output}")


if __name__ == "__main__":
    main()
