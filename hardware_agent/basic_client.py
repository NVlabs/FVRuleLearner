########################################################
# NVIDIA License
#
# 1. Definitions
#
# “Licensor” means any person or entity that distributes its Work.
# “Work” means (a) the original work of authorship made available under this license, which may include software, documentation, or other files, and (b) any additions to or derivative works  thereof  that are made available under this license.
# The terms “reproduce,” “reproduction,” “derivative works,” and “distribution” have the meaning as provided under U.S. copyright law; provided, however, that for the purposes of this license, derivative works shall not include works that remain separable from, or merely link (or bind by name) to the interfaces of, the Work.
# Works are “made available” under this license by including in or with the Work either (a) a copyright notice referencing the applicability of this license to the Work, or (b) a copy of this license.
#
# 2. License Grant
#
# 2.1 Copyright Grant. Subject to the terms and conditions of this license, each Licensor grants to you a perpetual, worldwide, non-exclusive, royalty-free, copyright license to use, reproduce, prepare derivative works of, publicly display, publicly perform, sublicense and distribute its Work and any resulting derivative works in any form.
#
# 3. Limitations
#
# 3.1 Redistribution. You may reproduce or distribute the Work only if (a) you do so under this license, (b) you include a complete copy of this license with your distribution, and (c) you retain without modification any copyright, patent, trademark, or attribution notices that are present in the Work.
#
# 3.2 Derivative Works. You may specify that additional or different terms apply to the use, reproduction, and distribution of your derivative works of the Work (“Your Terms”) only if (a) Your Terms provide that the use limitation in Section 3.3 applies to your derivative works, and (b) you identify the specific derivative works that are subject to Your Terms. Notwithstanding Your Terms, this license (including the redistribution requirements in Section 3.1) will continue to apply to the Work itself.
#
# 3.3 Use Limitation. The Work and any derivative works thereof only may be used or intended for use non-commercially. Notwithstanding the foregoing, NVIDIA Corporation and its affiliates may use the Work and any derivative works commercially. As used herein, “non-commercially” means for research or evaluation purposes only.
#
# 3.4 Patent Claims. If you bring or threaten to bring a patent claim against any Licensor (including any claim, cross-claim or counterclaim in a lawsuit) to enforce any patents that you allege are infringed by any Work, then your rights under this license from such Licensor (including the grant in Section 2.1) will terminate immediately.
#
# 3.5 Trademarks. This license does not grant any rights to use any Licensor’s or its affiliates’ names, logos, or trademarks, except as necessary to reproduce the notices described in this license.
#
# 3.6 Termination. If you violate any term of this license, then your rights under this license (including the grant in Section 2.1) will terminate immediately.
#
# 4. Disclaimer of Warranty.
#
# THE WORK IS PROVIDED “AS IS” WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WARRANTIES OR CONDITIONS OF 
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR NON-INFRINGEMENT. YOU BEAR THE RISK OF UNDERTAKING ANY ACTIVITIES UNDER THIS LICENSE. 
#
# 5. Limitation of Liability.
#
# EXCEPT AS PROHIBITED BY APPLICABLE LAW, IN NO EVENT AND UNDER NO LEGAL THEORY, WHETHER IN TORT (INCLUDING NEGLIGENCE), CONTRACT, OR OTHERWISE SHALL ANY LICENSOR BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR RELATED TO THIS LICENSE, THE USE OR INABILITY TO USE THE WORK (INCLUDING BUT NOT LIMITED TO LOSS OF GOODWILL, BUSINESS INTERRUPTION, LOST PROFITS OR DATA, COMPUTER FAILURE OR MALFUNCTION, OR ANY OTHER DAMAGES OR LOSSES), EVEN IF THE LICENSOR HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
########################################################
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