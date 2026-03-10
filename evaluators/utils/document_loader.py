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

"""Document loader for source documents and long-context files."""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

try:
    from markitdown import MarkItDown
    HAS_MARKITDOWN = True
except ImportError:
    HAS_MARKITDOWN = False


def normalize_title(text: str) -> str:
    """Normalize title: fullwidth-to-halfwidth conversion, whitespace cleanup, lowercase."""
    if not text:
        return ""
    
    result = ""
    for char in text:
        code = ord(char)
        if 0xFF01 <= code <= 0xFF5E:
            result += chr(code - 0xFEE0)
        elif code == 0x3000:
            result += ' '
        else:
            result += char
    
    result = result.replace('\u201c', '"').replace('\u201d', '"')
    result = result.replace('\u2018', "'").replace('\u2019', "'")
    result = result.replace('\u300c', '"').replace('\u300d', '"')
    result = result.replace('\u300e', '"').replace('\u300f', '"')
    
    result = re.sub(r'\s+', ' ', result).strip()
    result = result.lower()
    
    return result


def fuzzy_title_match(needle: str, haystack: str, threshold: float = 0.6) -> bool:
    """Fuzzy title matching with normalization, keyword overlap, and edit distance."""
    needle_norm = normalize_title(needle)
    haystack_norm = normalize_title(haystack)
    
    if not needle_norm or not haystack_norm:
        return False
    
    if needle_norm == haystack_norm:
        return True
    
    if needle_norm in haystack_norm or haystack_norm in needle_norm:
        return True
    
    def remove_ellipsis(s: str) -> str:
        s = re.sub(r'\.{2,}|\u2026+|\u3002{2,}', '', s)
        s = re.sub(r'[,\uff0c.\u3002!\uff01?\uff1f;\uff1b:\uff1a\-_\[\]\u3010\u3011()\uff08\uff09{}\u300c\u300d\u300e\u300f\u201c\u201d\'\']+', ' ', s)
        return re.sub(r'\s+', ' ', s).strip()
    
    needle_clean = remove_ellipsis(needle_norm)
    haystack_clean = remove_ellipsis(haystack_norm)
    
    if needle_clean and haystack_clean:
        if needle_clean == haystack_clean:
            return True
        if needle_clean in haystack_clean or haystack_clean in needle_clean:
            return True
    
    stopwords = {'\u7684', '\u4e86', '\u662f', '\u5728', '\u548c', '\u4e0e', '\u8fd9', '\u90a3', '\u6709', '\u4e3a', '\u4ee5', '\u53ca',
                 'the', 'a', 'an', 'of', 'to', 'in', 'for', 'on', 'with', 'and', 'or', 'is', 'are',
                 'video', 'audio', 'image', 'doc', 'document', 'file', 'mp4', 'mp3', 'pdf', 'jpg', 'png'}
    
    def tokenize(s: str) -> List[str]:
        tokens = []
        words = re.findall(r'[a-zA-Z]+', s)
        tokens.extend([w.lower() for w in words if len(w) > 1 and w.lower() not in stopwords])
        chinese_chars = re.findall(r'[\u4e00-\u9fff]+', s)
        for chars in chinese_chars:
            if len(chars) >= 2:
                tokens.append(chars)
            for c in chars:
                if c not in stopwords:
                    tokens.append(c)
        return tokens
    
    needle_tokens = tokenize(needle_norm)
    haystack_tokens = tokenize(haystack_norm)
    
    if len(needle_tokens) >= 2:
        matched = sum(1 for t in needle_tokens if t in haystack_norm or any(t in ht for ht in haystack_tokens))
        if matched / len(needle_tokens) >= threshold:
            return True
    
    brand_keywords = {
        'bbc', 'ted', 'cnn', 'nbc', 'abc', 'cbs', 'fox', 'npr', 'pbs',
        'youtube', 'vimeo', 'bilibili', 'netflix', 'hbo', 'disney',
        'national geographic', 'discovery', 'history channel',
        'mit', 'harvard', 'stanford', 'oxford', 'cambridge',
        'google', 'microsoft', 'apple', 'amazon', 'facebook', 'meta',
        'openai', 'anthropic', 'deepmind',
    }
    
    needle_words = set(re.findall(r'[a-zA-Z]+', needle_norm.lower()))
    haystack_words_str = haystack_norm.lower()
    
    for brand in brand_keywords:
        if brand in needle_words or brand in needle_norm.lower():
            if brand in haystack_words_str:
                return True
    
    def levenshtein_ratio(s1: str, s2: str) -> float:
        if not s1 or not s2:
            return 0.0
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        
        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        
        max_len = max(len(s1), len(s2))
        return 1 - distances[-1] / max_len if max_len > 0 else 0.0
    
    if len(needle_clean) >= 5 and len(haystack_clean) >= 5:
        ratio = levenshtein_ratio(needle_clean, haystack_clean)
        if ratio >= 0.75:
            return True
    
    return False


class DocumentLoader:
    """Loads and manages source documents and long-context data."""
    
    def __init__(self):
        self.source_documents: Dict[str, str] = {}
        self.long_context: List[Dict] = []
        self._title_to_idx: Dict[str, int] = {}
    
    def load_json(self, path: Path) -> Any:
        return json.loads(path.read_text(encoding='utf-8'))
    
    def load_long_context(self, path: Path) -> None:
        data = self.load_json(path)
        self.long_context = data if isinstance(data, list) else [data] if data else []
        self._build_index()
    
    def _build_index(self) -> None:
        self._title_to_idx.clear()
        for idx, item in enumerate(self.long_context):
            if isinstance(item, dict):
                title = item.get('title', '').lower().strip()
                if title:
                    self._title_to_idx[title] = idx
    
    def load_source_folder(self, source_folder: Path) -> None:
        if not source_folder.exists():
            return
        
        source_subfolder = source_folder / "source"
        if source_subfolder.exists() and source_subfolder.is_dir():
            source_folder = source_subfolder
        
        for file_path in source_folder.iterdir():
            if file_path.is_file() and not file_path.name.startswith('.'):
                self._load_single_file(file_path)
    
    def _load_single_file(self, file_path: Path) -> None:
        suffix = file_path.suffix.lower()
        
        if suffix in {".txt", ".md", ".json"}:
            try:
                content = file_path.read_text(encoding="utf-8")
                self.source_documents[file_path.name] = content
            except Exception as e:
                self.source_documents[file_path.name] = f"[Text file unreadable: {file_path.name}, error: {e}]"
            return
        
        if suffix in {".doc", ".docx", ".pdf", ".ppt", ".pptx", ".html", ".htm"}:
            if HAS_MARKITDOWN:
                try:
                    md = MarkItDown()
                    result = md.convert(str(file_path))
                    text = getattr(result, "text", None) or str(result)
                    self.source_documents[file_path.name] = text
                except Exception as e:
                    self.source_documents[file_path.name] = f"[Binary file unreadable: {file_path.name}, error: {e}]"
            else:
                self.source_documents[file_path.name] = f"[Binary file (install markitdown to parse): {file_path.name}]"
            return
        
        try:
            content = file_path.read_text(encoding="utf-8")
            self.source_documents[file_path.name] = content
        except Exception:
            self.source_documents[file_path.name] = f"[Binary file: {file_path.name}]"
    
    def get_content_by_index(self, idx: int) -> Tuple[Optional[str], Optional[str]]:
        if 0 <= idx < len(self.long_context):
            item = self.long_context[idx]
            if isinstance(item, dict):
                return item.get('page_body', ''), item.get('title', f'long_context[{idx}]')
        return None, None
    
    def get_content_by_title(self, title: str) -> Tuple[Optional[str], Optional[str]]:
        """Get long-context content by title with fuzzy matching."""
        title_norm = normalize_title(title)
        
        for idx, item in enumerate(self.long_context):
            if isinstance(item, dict):
                item_title = item.get('title', '')
                if normalize_title(item_title) == title_norm:
                    return item.get('page_body', ''), item.get('title', f'long_context[{idx}]')
        
        for idx, item in enumerate(self.long_context):
            if isinstance(item, dict):
                item_title = item.get('title', '')
                if fuzzy_title_match(title, item_title):
                    return item.get('page_body', ''), item.get('title', f'long_context[{idx}]')
        
        return None, None
    
    def get_source_document(self, filename: str) -> Optional[str]:
        return self.source_documents.get(filename)
