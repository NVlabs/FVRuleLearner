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
from typing import Dict, List, Literal, Optional, Union

from openai import OpenAI


class BasicClient:
    def __init__(
        self,
        llm_config: Optional[Union[Dict, Literal[False]]],
    ):
        self._llm_config = llm_config
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.system_message = None
        self.client = None
        self.model_name = None
        self.temperature = None
        self._validate_llm_config(llm_config)

    def _validate_llm_config(self, llm_config):
        assert llm_config in (None, False) or isinstance(
            llm_config, dict
        ), "llm_config must be a dict or False or None."

        if llm_config is False:
            self.client = None
            self.model_name = None
            self.temperature = None
            return

        self.llm_config = llm_config or {}
        config_list = self.llm_config.get("config_list", [])
        if not config_list or not isinstance(config_list[0], dict):
            raise ValueError("BasicClient expects a non-empty config_list.")

        config = config_list[0]
        self.model_name = config.get("model")
        self.temperature = self.llm_config.get("temperature")
        base_url = config.get("base_url")
        api_key = config.get("api_key")

        if not self.model_name:
            raise ValueError("BasicClient requires a model name.")

        if not base_url and config.get("gateway_chat_type"):
            raise RuntimeError(
                "Legacy gateway_chat_type configs are not supported in release mode; use OpenAI-compatible configs."
            )

        client_kwargs = {}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        timeout = self.llm_config.get("timeout")
        if timeout is not None:
            client_kwargs["timeout"] = timeout

        self.client = OpenAI(**client_kwargs)

    def resync_client(self):
        self._validate_llm_config(self._llm_config)

    def ask_client(
        self,
        prompt: str,
        is_extract_response: bool = True,
        system_message: str | None = None,
    ) -> str:
        self.resync_client()

        sm = system_message if system_message is not None else self.system_message
        messages = []
        if sm:
            messages.append({"role": "system", "content": sm})
        messages.append({"role": "user", "content": prompt})
        return self.create(messages=messages, is_extract_response=is_extract_response)

    def create(
        self,
        messages: List[Dict],
        is_extract_response: bool = True,
        cancellation_token=None,
    ) -> str:
        del cancellation_token
        self.resync_client()

        create_kwargs = {
            "model": self.model_name,
            "messages": messages,
        }
        if self.temperature is not None and not str(self.model_name).startswith(("o1", "o3", "o4")):
            create_kwargs["temperature"] = self.temperature

        response = self.client.chat.completions.create(**create_kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0

        if is_extract_response:
            return response.choices[0].message.content
        return response

    def chat(self, prompt: str, n=1, stop=None) -> list:
        del stop
        outputs = []
        while n > 0:
            cnt = min(n, 20)
            n -= cnt
            res = self.ask_client(prompt=prompt, is_extract_response=False)
            outputs.extend([choice.message.content for choice in res.choices[:cnt]])
        return outputs

    def gpt_usage(self):
        return {
            "completion_tokens": self.completion_tokens,
            "prompt_tokens": self.prompt_tokens,
            "cost": None,
        }
