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
Folder Processor Module

This module provides functionality to process all files in a folder
and prepare them for multi-modal LLM processing.

Supports:
- Images: jpg, jpeg, png, gif, webp
- Videos: mp4, avi, mov, mkv, webm, flv, wmv, m4v
- Audio: wav, mp3, m4a
- Documents: pdf, docx, doc, txt, xlsx, xls, pptx, ppt, html, htm
- Data: json, jsonld, csv
- Archives: zip

Usage:
    from src.io.folder_processor import process_folder_for_task
    
    task_content, task_description, multimodal_files = process_folder_for_task(
        folder_path="data/000",
        query="Please organize important references based on the content of this PDF file and images"
    )
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Import existing converters and utilities from input_handler
from .input_handler import (
    DocumentConverterResult,
    XlsxConverter,
    DocxConverter,
    HtmlConverter,
    PptxConverter,
    ZipConverter,
    process_input,
)

# Try to import optional dependencies
try:
    import pdfminer.high_level
    from pdfminer.pdfpage import PDFPage
    HAS_PDFMINER = True
except ImportError:
    HAS_PDFMINER = False

try:
    from markitdown import MarkItDown
    HAS_MARKITDOWN = True
except ImportError:
    HAS_MARKITDOWN = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

import json


# File type categories
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a"}
DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".html", ".htm", ".md", ".markdown", ".css"}
SPREADSHEET_EXTENSIONS = {".xlsx", ".xls", ".csv"}
PRESENTATION_EXTENSIONS = {".pptx", ".ppt"}
DATA_EXTENSIONS = {".jsonld", ".json", ".jsonl"}
ARCHIVE_EXTENSIONS = {".zip"}


@dataclass
class FileInfo:
    """Information about a single file."""
    path: str
    name: str
    extension: str
    category: str
    size_bytes: int
    
    @property
    def is_multimodal(self) -> bool:
        """Check if file requires multimodal processing (image/video/audio)."""
        return self.category in ["image", "video", "audio"]


@dataclass
class FolderContents:
    """Structured representation of folder contents."""
    folder_path: str
    files: List[FileInfo] = field(default_factory=list)
    
    @property
    def images(self) -> List[FileInfo]:
        return [f for f in self.files if f.category == "image"]
    
    @property
    def videos(self) -> List[FileInfo]:
        return [f for f in self.files if f.category == "video"]
    
    @property
    def audios(self) -> List[FileInfo]:
        return [f for f in self.files if f.category == "audio"]
    
    @property
    def documents(self) -> List[FileInfo]:
        return [f for f in self.files if f.category == "document"]
    
    @property
    def spreadsheets(self) -> List[FileInfo]:
        return [f for f in self.files if f.category == "spreadsheet"]
    
    @property
    def presentations(self) -> List[FileInfo]:
        return [f for f in self.files if f.category == "presentation"]
    
    @property
    def data_files(self) -> List[FileInfo]:
        return [f for f in self.files if f.category == "data"]
    
    @property
    def archives(self) -> List[FileInfo]:
        return [f for f in self.files if f.category == "archive"]
    
    @property
    def other_files(self) -> List[FileInfo]:
        return [f for f in self.files if f.category == "other"]
    
    @property
    def multimodal_files(self) -> List[FileInfo]:
        """Get all files that require multimodal processing."""
        return [f for f in self.files if f.is_multimodal]
    
    @property
    def text_extractable_files(self) -> List[FileInfo]:
        """Get all files that can have text extracted."""
        return [f for f in self.files if f.category in 
                ["document", "spreadsheet", "presentation", "data"]]
    
    def get_summary(self) -> str:
        """Get a summary of folder contents."""
        summary_parts = [f"Folder: {self.folder_path}"]
        summary_parts.append(f"Total files: {len(self.files)}")
        
        categories = {}
        for f in self.files:
            categories[f.category] = categories.get(f.category, 0) + 1
        
        for cat, count in sorted(categories.items()):
            summary_parts.append(f"  - {cat}: {count}")
        
        return "\n".join(summary_parts)


def get_file_category(extension: str) -> str:
    """Determine the category of a file based on its extension."""
    ext = extension.lower()
    
    if ext in IMAGE_EXTENSIONS:
        return "image"
    elif ext in VIDEO_EXTENSIONS:
        return "video"
    elif ext in AUDIO_EXTENSIONS:
        return "audio"
    elif ext in DOCUMENT_EXTENSIONS:
        return "document"
    elif ext in SPREADSHEET_EXTENSIONS:
        return "spreadsheet"
    elif ext in PRESENTATION_EXTENSIONS:
        return "presentation"
    elif ext in DATA_EXTENSIONS:
        return "data"
    elif ext in ARCHIVE_EXTENSIONS:
        return "archive"
    else:
        return "other"


def scan_folder(folder_path: str, recursive: bool = False) -> FolderContents:
    """
    Scan a folder and categorize all files.
    
    Args:
        folder_path: Path to the folder to scan
        recursive: Whether to scan subdirectories recursively
        
    Returns:
        FolderContents object with categorized files
    """
    folder_path = os.path.abspath(folder_path)
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Path is not a directory: {folder_path}")
    
    contents = FolderContents(folder_path=folder_path)
    
    if recursive:
        for root, _, files in os.walk(folder_path):
            for filename in files:
                if filename.startswith("."):  # Skip hidden files
                    continue
                file_path = os.path.join(root, filename)
                _add_file_info(contents, file_path, filename)
    else:
        for filename in os.listdir(folder_path):
            if filename.startswith("."):  # Skip hidden files
                continue
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                _add_file_info(contents, file_path, filename)
    
    return contents


def _add_file_info(contents: FolderContents, file_path: str, filename: str) -> None:
    """Add file information to FolderContents."""
    _, ext = os.path.splitext(filename)
    category = get_file_category(ext)
    
    try:
        size = os.path.getsize(file_path)
    except OSError:
        size = 0
    
    contents.files.append(FileInfo(
        path=file_path,
        name=filename,
        extension=ext.lower(),
        category=category,
        size_bytes=size
    ))


def _extract_file_content(file_info: FileInfo, max_content_length: int = 200_000) -> Optional[str]:
    """
    Extract text content from a file using existing converters.
    
    Args:
        file_info: FileInfo object for the file
        max_content_length: Maximum length of content to return
        
    Returns:
        Extracted text content or None if extraction failed
    """
    file_path = file_info.path
    ext = file_info.extension.lower()
    
    try:
        parsing_result = None
        
        # Use existing converters from input_handler
        if ext == ".txt" or ext in [".md", ".markdown", ".css"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            parsing_result = DocumentConverterResult(title=None, text_content=content)
        
        elif ext in [".json", ".jsonld", ".jsonl"]:
            # Skip all JSON/JSONL files; long context handled by RAG tools
            return None
        
        elif ext in [".xlsx", ".xls"]:
            # Agent reads Excel via tools
            return None
        
        elif ext == ".pdf":
            # Agent reads PDF via tools
            return None
        
        elif ext in [".docx", ".doc"]:
            # Agent reads Word via tools
            return None
        
        elif ext in [".html", ".htm"]:
            return None
        
        elif ext in [".pptx", ".ppt"]:
            # Agent reads PPT via tools
            return None
        
        elif ext == ".zip":
            return None
        
        elif ext == ".csv":
            # Agent reads CSV via tools
            return None
        
        # Try MarkItDown as fallback for other file types
        if parsing_result is None and HAS_MARKITDOWN:
            try:
                md = MarkItDown(enable_plugins=True)
                parsing_result = md.convert(file_path)
            except Exception:
                pass
        
        # Extract content from result
        if parsing_result:
            content = parsing_result.text_content
            if content and len(content) > max_content_length:
                content = content[:max_content_length] + "\n... [Content truncated]"
            return content
        
    except Exception as e:
        return f"[Error extracting content: {str(e)}]"
    
    return None


def _get_image_info(file_info: FileInfo) -> str:
    """Get image information string."""
    info_parts = [f"Image file: {file_info.name}"]
    info_parts.append(f"Path: {file_info.path}")
    
    ext = file_info.extension.lower()
    
    # SVG files need special handling
    if ext == ".svg":
        info_parts.append("Format: SVG (Scalable Vector Graphics)")
        try:
            # Try to get SVG dimensions from file content
            with open(file_info.path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(2000)  # Read first 2000 chars
                import re
                # Try to find width/height attributes
                width_match = re.search(r'width=["\']?(\d+)', content)
                height_match = re.search(r'height=["\']?(\d+)', content)
                if width_match and height_match:
                    info_parts.append(f"Dimensions: {width_match.group(1)}x{height_match.group(1)} (from attributes)")
                # Try to find viewBox
                viewbox_match = re.search(r'viewBox=["\']?[\d\s]+\s+[\d\s]+\s+(\d+)\s+(\d+)', content)
                if viewbox_match:
                    info_parts.append(f"ViewBox: {viewbox_match.group(1)}x{viewbox_match.group(2)}")
        except Exception:
            pass
    elif HAS_PIL:
        try:
            with Image.open(file_info.path) as img:
                width, height = img.size
                info_parts.append(f"Dimensions: {width}x{height} pixels")
                info_parts.append(f"Format: {img.format}")
        except Exception:
            pass
    
    return "\n".join(info_parts)


def _get_video_info(file_info: FileInfo) -> str:
    """Get video information string."""
    info_parts = [f"Video file: {file_info.name}"]
    info_parts.append(f"Path: {file_info.path}")
    
    try:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(file_info.path)
        info_parts.append(f"Duration: {clip.duration:.2f} seconds")
        info_parts.append(f"Resolution: {clip.w}x{clip.h}")
        info_parts.append(f"FPS: {clip.fps:.1f}")
        clip.close()
    except Exception:
        try:
            import cv2
            cap = cv2.VideoCapture(file_info.path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if fps > 0:
                duration = frame_count / fps
                info_parts.append(f"Duration: {duration:.2f} seconds")
            info_parts.append(f"Resolution: {width}x{height}")
            info_parts.append(f"FPS: {fps:.1f}")
            cap.release()
        except Exception:
            pass
    
    return "\n".join(info_parts)


def _get_audio_info(file_info: FileInfo) -> str:
    """Get audio information string."""
    info_parts = [f"Audio file: {file_info.name}"]
    info_parts.append(f"Path: {file_info.path}")
    
    ext = file_info.extension.lower()
    
    if ext == ".wav":
        try:
            import wave
            with wave.open(file_info.path, "rb") as audio_file:
                duration = audio_file.getnframes() / float(audio_file.getframerate())
                sample_rate = audio_file.getframerate()
                channels = audio_file.getnchannels()
                info_parts.append(f"Duration: {duration:.2f} seconds")
                info_parts.append(f"Sample rate: {sample_rate} Hz")
                info_parts.append(f"Channels: {channels}")
        except Exception:
            pass
    else:
        try:
            from mutagen import File as MutagenFile
            audio = MutagenFile(file_info.path)
            if audio and hasattr(audio, "info") and hasattr(audio.info, "length"):
                info_parts.append(f"Duration: {audio.info.length:.2f} seconds")
                if hasattr(audio.info, "sample_rate"):
                    info_parts.append(f"Sample rate: {audio.info.sample_rate} Hz")
        except Exception:
            pass
    
    return "\n".join(info_parts)


def process_folder_for_task(
    folder_path: str,
    query: str,
    recursive: bool = False,
    include_file_contents: bool = True,
    max_content_length: int = 200_000,
    target_db_path: str = None
) -> Tuple[str, str, List[str]]:
    """
    Process all files in a folder and prepare task description for LLM.
    
    This function:
    1. Scans the folder and categorizes all files
    2. Extracts text content from documents, spreadsheets, etc.
    3. Prepares multimodal file information (images, videos, audio)
    4. Generates a comprehensive task description with tool usage guidance
    
    Args:
        folder_path: Path to the folder to process
        query: The user's query/question about the folder contents
        recursive: Whether to scan subdirectories recursively
        include_file_contents: Whether to include extracted file contents
        max_content_length: Maximum length of content per file
        target_db_path: Optional path to a specific .chunks.db file to use.
                       If provided, ONLY this db file will be shown in the task description.
                       Other db files will be completely hidden from the model.
                       This ensures the model can only use the specified context size.
        
    Returns:
        Tuple of:
        - task_content: Full content string for LLM (includes file contents)
        - task_description: Task description with tool guidance
        - multimodal_files: List of paths to multimodal files (images, videos, audio)
    """
    # Scan folder
    contents = scan_folder(folder_path, recursive=recursive)
    
    # If target_db_path is specified, filter out ALL other db files AND ALL long_context*.json files
    # This ensures the model can ONLY see and use the specified db file
    # The model should NOT know that other context sizes exist
    if target_db_path:
        # Normalize target_db_path to absolute path for comparison
        # (scan_folder converts all paths to absolute paths)
        target_db_path_abs = os.path.abspath(target_db_path)
        
        # Get the base name of the target db (e.g., "long_context_sampled_32k.json" from "long_context_sampled_32k.json.chunks.db")
        target_db_name = os.path.basename(target_db_path)
        target_json_name = target_db_name.replace('.chunks.db', '') if target_db_name.endswith('.chunks.db') else None
        
        # Remove ALL .chunks.db files except the target
        # Remove ALL long_context*.json files (model should only use the db file directly)
        def should_keep_file(f):
            # Remove all .chunks.db files except the target
            if f.name.endswith('.chunks.db'):
                return f.path == target_db_path_abs
            # Remove ALL long_context*.json files - model should use db file directly
            # This includes: long_context.json, long_context_sampled_32k.json, etc.
            if 'long_context' in f.name.lower() and f.extension.lower() in ['.json', '.jsonl']:
                return False  # Always remove, model should use db file
            return True
        
        contents.files = [f for f in contents.files if should_keep_file(f)]
        
        # Do NOT add the target db file to the file list
        # The model will only know about it through the RAG tools guidance
    
    # Build task description
    task_parts = []
    task_parts.append(f"# Task\n\n{query}\n")
    
    # Add folder summary
    task_parts.append(f"\n## Folder Contents Summary\n\n{contents.get_summary()}\n")
    
    # Process user-uploaded files (excluding all JSON files)
    # JSON files include: long_context.json, useful_search.json, noise_search.json, etc.
    # Exclude all JSON/JSONL files
    local_doc_files = [f for f in contents.text_extractable_files 
                       if f.extension.lower() not in [".json", ".jsonld", ".jsonl"]]
    
    # Also include presentations
    local_doc_files.extend(contents.presentations)
    
    if local_doc_files:
        task_parts.append("\n## User Uploaded Files\n")
        
        # Separate files that can be directly included vs those that need tools
        direct_include_files = [f for f in local_doc_files if f.extension.lower() in [".txt", ".md", ".markdown"]]
        tool_required_files = [f for f in local_doc_files if f.extension.lower() not in [".txt", ".md", ".markdown"]]
        
        # For .txt and .md files, include content directly in prompt
        if direct_include_files:
            task_parts.append("\n### 📄 User Document Content (Included Directly Below)\n")
            task_parts.append("**⚠️ IMPORTANT: The following are core documents uploaded by the user. Content is included directly. Please read carefully and cite these in your report.**\n")
            
            for file_info in direct_include_files:
                content = _extract_file_content(file_info, max_content_length)
                if content:
                    task_parts.append(f"\n#### 📖 {file_info.name}\n")
                    task_parts.append(f"**File Path**: `{file_info.path}`\n")
                    task_parts.append(f"**File Size**: {file_info.size_bytes / 1024:.1f} KB\n")
                    task_parts.append("\n**File Content**:\n")
                    task_parts.append("```markdown")
                    task_parts.append(content)
                    task_parts.append("```\n")
                else:
                    task_parts.append(f"\n#### {file_info.name}\n")
                    task_parts.append(f"Path: `{file_info.path}` (Unable to extract content)\n")
        
        # For other files (PDF, Excel, etc.), list them and provide tool guidance
        if tool_required_files:
            task_parts.append("\n### 📁 Files Requiring Tool Access\n")
            task_parts.append("\n**⚠️ IMPORTANT: The following files require tools to read their content.**")
            task_parts.append("**These files are crucial for completing the task. Please use the `tool-reading` server tools to read them.**\n")
            
            task_parts.append("\n| Filename | Type | Size | Path |")
            task_parts.append("| --- | --- | --- | --- |")
            
            for file_info in tool_required_files:
                size_str = f"{file_info.size_bytes / 1024:.1f} KB" if file_info.size_bytes < 1024 * 1024 else f"{file_info.size_bytes / 1024 / 1024:.1f} MB"
                task_parts.append(f"| {file_info.name} | {file_info.extension} | {size_str} | `{file_info.path}` |")
            
            task_parts.append("\n\n**📖 How to Read These Files (using tool-reading server):**")
            task_parts.append("- `read_pdf_pages(file_path, start_page, end_page)` - Read specific PDF pages (max 3 pages per call)")
            task_parts.append("- `read_excel_rows(file_path, start_row, end_row)` - Read specific Excel/CSV rows")
            task_parts.append("- `search_in_file(file_path, keyword)` - Search for keywords in file (recommended to locate content first)")
            task_parts.append("- `get_file_info(file_path)` - Get file structure information")
            task_parts.append("- `convert_to_markdown(uri)` - Convert Word/PPT/HTML to Markdown (uri format: file:///path/to/file)\n")
    
    # Process multimodal files
    multimodal_files = []
    
    # Images
    if contents.images:
        task_parts.append("\n## Image Files\n")
        task_parts.append("\nThe following image files are available for analysis:\n")
        
        for file_info in contents.images:
            multimodal_files.append(file_info.path)
            task_parts.append(f"\n### {file_info.name}\n")
            task_parts.append(_get_image_info(file_info))
        
        task_parts.append("\n\n**IMPORTANT**: Use the 'vision_understanding_advanced' tool to analyze these images.")
        task_parts.append("This tool provides multi-turn verification, confidence scoring, and cross-validation.")
        task_parts.append("Recommended approach:")
        task_parts.append("1. Call vision_understanding_advanced with a specific question about each image")
        task_parts.append("2. Review the confidence score and metadata")
        task_parts.append("3. If confidence < 0.75, use follow-up analysis or web search for verification\n")
    
    # Videos
    if contents.videos:
        task_parts.append("\n## Video Files\n")
        task_parts.append("\nThe following video files are available for analysis:\n")
        
        for file_info in contents.videos:
            multimodal_files.append(file_info.path)
            task_parts.append(f"\n### {file_info.name}\n")
            task_parts.append(_get_video_info(file_info))
        
        task_parts.append("\n\n**IMPORTANT**: Use the 'video_understanding_advanced' tool to analyze these videos.")
        task_parts.append("Recommendation:")
        task_parts.append("- Use enable_verification=true for detailed action/scene analysis")
        task_parts.append("- For quick preview, use 'video_quick_analysis' tool")
        task_parts.append("- To analyze specific time ranges, use 'video_temporal_qa' with start_time and end_time")
        task_parts.append("- To extract key moments/frames, use 'video_extract_keyframes' tool\n")
    
    # Audio
    if contents.audios:
        task_parts.append("\n## Audio Files\n")
        task_parts.append("\nThe following audio files are available for analysis:\n")
        
        for file_info in contents.audios:
            multimodal_files.append(file_info.path)
            task_parts.append(f"\n### {file_info.name}\n")
            task_parts.append(_get_audio_info(file_info))
        
        task_parts.append("\n\n**IMPORTANT**: Use the 'audio_understanding_advanced' tool to analyze these audio files.")
        task_parts.append("Recommendation:")
        task_parts.append("- Use enable_verification=true for critical transcriptions")
        task_parts.append("- For quick transcription, use 'audio_quick_transcription' tool")
        task_parts.append("- To answer specific questions about the audio, use 'audio_question_answering_enhanced'\n")
    
    # Long context files (RAG)
    # When target_db_path is specified, we ONLY show that db file
    # When not specified, we look for db files or json files as fallback
    
    # Determine the recommended db path
    recommended_db_path = None
    if target_db_path:
        # Use the specified target_db_path exclusively
        recommended_db_path = os.path.abspath(target_db_path)
    else:
        # Check for .db files (pre-built embedding databases)
        db_files = [f for f in contents.other_files if f.name.endswith('.chunks.db')]
        if db_files:
            # Use the smallest db file as default
            db_files_sorted = sorted(db_files, key=lambda f: f.size_bytes)
            recommended_db_path = db_files_sorted[0].path
        else:
            # Fallback: check for .json files and their corresponding db files
            long_context_files = [f for f in contents.data_files if "long_context" in f.name.lower()]
            for file_info in long_context_files:
                db_path = file_info.path + ".chunks.db"
                if os.path.exists(db_path):
                    recommended_db_path = db_path
                    break
    
    # Only show RAG section if we have a recommended db path
    if recommended_db_path:
        task_parts.append("\n## Long Context Documents (RAG - Task-Specific Knowledge Base)\n")
        task_parts.append("\n**🔍 What is Long Context?**")
        task_parts.append("Long Context is a **task-specific knowledge base** containing **pre-retrieved web materials** that we have gathered specifically for the current task.")
        task_parts.append("These materials have been pre-processed and stored in a database, and can be quickly retrieved via RAG tools.\n")
        task_parts.append("**⚠️ IMPORTANT:**")
        task_parts.append("- Long Context is **different from** the user-uploaded files (PPT, PDF, etc.) above")
        task_parts.append("- **User-uploaded files**: Original materials directly provided by the user")
        task_parts.append("- **Long Context**: Supplementary reference materials we pre-retrieved (stored in database)\n")
        task_parts.append("**🚀 STRONGLY RECOMMENDED: Use RAG tools extensively to retrieve useful information from Long Context!**")
        task_parts.append("These materials are specifically prepared for the current task and may contain key information to solve the problem.\n")
        
        task_parts.append("\n**HOW TO USE RAG TOOLS**:")
        task_parts.append("- `rag_search`: Semantic search to find relevant passages")
        task_parts.append("- `rag_get_context`: Get concatenated context for answering questions")
        task_parts.append("- `rag_document_stats`: Get document collection statistics")
        task_parts.append("\n**⚠️ IMPORTANT DISTINCTION**:")
        task_parts.append("- **LOCAL FILES** (PPT, PDF above): Content is ALREADY in this prompt. Cite as [Doc: filename.ext]")
        task_parts.append("- **RAG DOCUMENTS**: Need to be searched. Cite as [long_context: \"title\", chunk N]")
        task_parts.append("\n**You should use BOTH sources** - local files for primary content, RAG for supplementary research.")
        task_parts.append(f"\n**⚠️ CRITICAL: When calling RAG tools, use this json_path: {recommended_db_path}**")
        task_parts.append("This database has pre-built embeddings and will load instantly.\n")
    
    # Other files
    other_files = [f for f in contents.other_files if "long_context" not in f.name.lower()]
    if other_files:
        task_parts.append("\n## Other Files\n")
        for file_info in other_files:
            task_parts.append(f"- {file_info.name} ({file_info.extension})\n")
    
    # Add output format requirement
    use_cn_prompt = os.environ.get("USE_CN_PROMPT", "0")
    if use_cn_prompt == "1":
        task_parts.append("\nPlease solve the given problem through task decomposition and MCP tool calls. **Generate the complete report content without wrapping in \\boxed{}.**")
    else:
        task_parts.append("\nYou should follow the format instruction in the request strictly. Generate the complete report content without wrapping it in \\boxed{}.")
    
    task_content = "\n".join(task_parts)
    task_description = task_content
    
    return task_content, task_description, multimodal_files


def process_folder_batch(
    folder_paths: List[str],
    query: str,
    recursive: bool = False,
    include_file_contents: bool = True,
    max_content_length: int = 200_000
) -> List[Tuple[str, str, str, List[str]]]:
    """
    Process multiple folders in batch.
    
    Args:
        folder_paths: List of folder paths to process
        query: The user's query/question about the folder contents
        recursive: Whether to scan subdirectories recursively
        include_file_contents: Whether to include extracted file contents
        max_content_length: Maximum length of content per file
        
    Returns:
        List of tuples, each containing:
        - folder_path: The original folder path
        - task_content: Full content string for LLM
        - task_description: Task description with tool guidance
        - multimodal_files: List of paths to multimodal files
    """
    results = []
    
    for folder_path in folder_paths:
        try:
            task_content, task_description, multimodal_files = process_folder_for_task(
                folder_path=folder_path,
                query=query,
                recursive=recursive,
                include_file_contents=include_file_contents,
                max_content_length=max_content_length
            )
            results.append((folder_path, task_content, task_description, multimodal_files))
        except Exception as e:
            error_msg = f"Error processing folder {folder_path}: {str(e)}"
            results.append((folder_path, error_msg, error_msg, []))
    
    return results
