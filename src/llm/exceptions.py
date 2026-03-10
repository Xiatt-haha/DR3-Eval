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
Custom exceptions for LLM clients.
"""

import re
from typing import Optional

def extract_trace_id(error: Exception) -> Optional[str]:
    """
    Extract trace ID from an exception.
    
    Looks for trace ID in:
    1. Response headers (eagleeye-traceid)
    2. Error message body (traceId: xxx)
    3. Error body dict
    
    Args:
        error: The exception to extract trace ID from
        
    Returns:
        The trace ID if found, None otherwise
    """
    trace_id = None
    
    # Try to get from response headers
    if hasattr(error, 'response') and error.response is not None:
        response = error.response
        if hasattr(response, 'headers'):
            trace_id = response.headers.get('eagleeye-traceid')
            if trace_id:
                return trace_id
            # Check for other common trace ID headers
            for header_name in ['x-request-id', 'x-trace-id', 'traceparent', 'x-amzn-requestid']:
                trace_id = response.headers.get(header_name)
                if trace_id:
                    return trace_id
    
    # Try to get from error body
    if hasattr(error, 'body') and error.body is not None:
        body = error.body
        if isinstance(body, dict):
            # Check detailMessage field
            detail_message = body.get('detailMessage', '')
            if detail_message:
                # Look for "traceId: xxx" pattern
                match = re.search(r'traceId:\s*([a-zA-Z0-9]+)', detail_message)
                if match:
                    return match.group(1)
            # Check for direct trace_id field
            trace_id = body.get('trace_id') or body.get('traceId') or body.get('request_id')
            if trace_id:
                return trace_id
    
    # Try to get from error message string
    error_str = str(error)
    match = re.search(r'traceId:\s*([a-zA-Z0-9]+)', error_str)
    if match:
        return match.group(1)
    
    # Try to get from eagleeye-traceid in error string
    match = re.search(r'eagleeye-traceid[:\s]+([a-zA-Z0-9]+)', error_str, re.IGNORECASE)
    if match:
        return match.group(1)
    
    return None

class APIConnectionSkipError(Exception):
    """
    Exception raised when an API connection error occurs and the current task should be skipped.
    
    This exception is NOT retried - it immediately stops the current task and allows
    the batch processor to move on to the next task.
    
    Common causes:
    - Rate limiting (429 errors)
    - Connection timeout
    - Server unavailable
    - Network issues
    
    Attributes:
        message: Error message
        original_error: The original exception that caused this error
        trace_id: The trace ID from the API response (for debugging with ops team)
    """
    
    def __init__(self, message: str, original_error: Exception = None, trace_id: str = None):
        self.message = message
        self.original_error = original_error
        # Try to extract trace_id from original error if not provided
        if trace_id is None and original_error is not None:
            trace_id = extract_trace_id(original_error)
        self.trace_id = trace_id
        super().__init__(self.message)
    
    def __str__(self):
        parts = [self.message]
        if self.trace_id:
            parts.append(f"TraceID: {self.trace_id}")
        if self.original_error:
            parts.append(f"Original error: {type(self.original_error).__name__}: {self.original_error}")
        return " | ".join(parts)
    
    def get_trace_id(self) -> Optional[str]:
        """Get the trace ID for this error."""
        return self.trace_id

class APIRateLimitError(APIConnectionSkipError):
    """
    Exception raised when rate limiting is detected.
    """
    pass

class APITimeoutError(APIConnectionSkipError):
    """
    Exception raised when API request times out.
    """
    pass
