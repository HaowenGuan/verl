# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import json
from typing import Any, Optional
from uuid import uuid4

from verl.utils.reward_score import vstar
from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class VideoSampleTool(BaseTool):
    """A tool for sampling frames from a video given the time window [start_time, end_time].

    - `get_openai_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "get_video_frames",
                "description": "A tool for extracting frames from a video clip given the time window [start_time, end_time] in seconds. This tool allows you to look at a specific segment of a video.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_time": {
                            "type": "float",
                            "description": "The start time of the video clip in seconds. Must be a float number with one decimal."
                        },
                        "end_time": {
                            "type": "float",
                            "description": "The start time of the video clip in seconds. Must be a float number with one decimal."
                        }
                    },
                    "required": ["start_time", "end_time"]
                }
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        if ground_truth is None:
            ground_truth = kwargs.get("create_kwargs", {}).get("ground_truth", None)
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        # start_time = parameters.get("start_time", None)
        # if not isinstance(start_time, str):
        #     start_time = str(start_time)
        # end_time = parameters.get("end_time", None)
        # if not isinstance(end_time, str):
        #     end_time = str(end_time)
        # self._instance_dict[instance_id]["response"] = f"[{start_time}, {end_time}]"

        answer = parameters.get("answer", None)
        if not isinstance(answer, str):
            answer = ""
        if "###" not in answer:
            answer = "### " + answer
        self._instance_dict[instance_id]["response"] = answer

        reward = await self.calc_reward(instance_id)

        return ToolResponse(text=tool_message, image=[]), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return vstar.compute_score(
            self._instance_dict[instance_id]["response"],
            self._instance_dict[instance_id]["ground_truth"]
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
