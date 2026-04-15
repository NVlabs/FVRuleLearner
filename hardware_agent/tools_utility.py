# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# utilities for tools descriptions
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union

def create_tool_tbl(tool_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    tool_table_map = {}
    for tool in tool_configs:
        tool_table_map[tool['name']] = tool
    return tool_table_map

def get_tools_descriptions(tools: List[str], tool_table_map: Dict[str, Any]) -> (str, str):
    tool_strings = []
    for name in tools:
        description = tool_table_map[name]['description']
        tool_strings.append(f"{name}: \n{description}")

    formatted_tools = "\n".join(tool_strings)
    tool_names = ", ".join([tool for tool in tools])
    return tool_names, formatted_tools