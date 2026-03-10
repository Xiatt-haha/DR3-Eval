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

from typing import Any


class ErrorBox:
    def __init__(self, error_msg: str):
        self.error_msg = error_msg

    def __str__(self):
        return self.error_msg

    @staticmethod
    def is_error_box(something):
        return isinstance(something, ErrorBox)


class ResponseBox:
    def __init__(self, response: Any, extra_info: dict = None):
        self.response = response
        self.extra_info = extra_info

    def __str__(self):
        return self.response

    @staticmethod
    def is_response_box(something):
        return isinstance(something, ResponseBox)

    def has_extra_info(self):
        return self.extra_info is not None

    def get_extra_info(self):
        return self.extra_info

    def get_response(self):
        return self.response
