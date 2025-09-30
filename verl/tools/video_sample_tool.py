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
import torch

from verl.utils.rollout_trace import rollout_trace_op
# from qwen_vl_utils import fetch_video
import torchvision.transforms.functional as TF

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse
from temp.vision_process import fetch_video

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
        self,
        instance_id: Optional[str] = None,
        video_path: Optional[str] = None,
        video_duration: Optional[float] = None,
        **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        if video_path is None:
            video_path = kwargs.get("create_kwargs", {}).get("video_path", None)
        if video_duration is None:
            video_duration = kwargs.get("create_kwargs", {}).get("video_duration", None)
        self._instance_dict[instance_id] = {
            "response": "",
            "video_path": video_path,
            "video_duration": round(video_duration, 1),
            "reward": 0.0,
            "remaining_turns": 4,
        }
        return instance_id, ToolResponse()

    async def extract_and_save_frames(
            self,
            instance_id: str,
            video_start: float,
            video_end: float,
            fps: float = None,
            nframes: int = 8,
            total_pixels: int = 20480 * 28 * 28,
            min_pixels: int = 16 * 28 * 28,
            max_pixels: int = 200 * 28 * 28,
    ):
        """Reads video, extracts frames using fetch_video, saves them, and returns paths."""
        video_path = self._instance_dict[instance_id]['video_path']
        try:
            # 1. Fetch the frame tensor using qwen_vl_utils
            # We construct the dictionary expected by fetch_video
            assert not (fps is not None and nframes is not None), "Only accept either `fps` or `nframes`"
            video_info_dict = {
                "type": "video",
                "video": f"file://{video_path}",  # Use file URI
                "nframes": nframes,
                "total_pixels": total_pixels,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "video_start": video_start,
                "video_end": video_end,
            }
            # fetch_video returns a tensor [n_frames, C, H, W] or list of PIL Images
            # Let's ensure it returns a tensor for consistent processing
            # Note: fetch_video already handles sampling and resizing based on parameters
            frame_data, sample_fps = fetch_video(video_info_dict, return_video_sample_fps=True)
            # Calculate the real sample_fps
            nframes = len(frame_data)
            frame_duration = 1 / sample_fps
            video_duration = frame_duration * nframes
            sample_fps = (nframes - 1) / video_duration

            if not isinstance(frame_data, torch.Tensor) or frame_data.ndim != 4:
                logger.error(f"fetch_video did not return expected tensor for {instance_id}. Got {type(frame_data)}")
                return None, None

            num_frames = frame_data.shape[0]
            if num_frames == 0:
                logger.warning(f"fetch_video returned 0 frames for {instance_id}. Skipping.")
                return None, None

            pil_images = []
            for i in range(num_frames):
                frame_tensor = frame_data[i]  # Shape [C, H, W]
                # Convert tensor to PIL Image (ensure it's in range 0-255 if needed)
                # fetch_video should return float tensor in [0,1], convert to uint8
                pil_images.append(TF.to_pil_image((frame_tensor * 255).byte()))

            return pil_images, sample_fps

        except Exception as e:
            logger.error(f"Error extracting/saving frames for {instance_id} from {video_path}: {e}", exc_info=True)
            return None, None

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        self._instance_dict[instance_id]["remaining_turns"] -= 1
        remaining_turns = self._instance_dict[instance_id]["remaining_turns"]
        if remaining_turns > 0:
            remaining_turn_message = f"\nYou have {remaining_turns} turns remaining to make additional tool calls."
        else:
            remaining_turn_message = "\nNo more tool call turns left. Please generate your final answer."

        try:
            start_time = parameters.get("start_time", None)
            if not isinstance(start_time, float):
                start_time = float(start_time)
            end_time = parameters.get("end_time", None)
            if not isinstance(end_time, float):
                end_time = float(end_time)
        except:
            message = f"[Invalid Tool Call] Fail to parse start_time or end_time:\n{parameters}"
            message += remaining_turn_message
            print(message)
            return ToolResponse(text=message), 0.0, {}
        # self._instance_dict[instance_id]["response"] = f"[{start_time}, {end_time}]"

        if start_time > end_time:
            start_time, end_time = end_time, start_time

        if end_time - start_time < 2.0:
            message = f"[Invalid Tool Call] Time window must be wider than 2.0 seconds."
            message += remaining_turn_message
            return ToolResponse(text=message), 0.0, {}

        if start_time < 0.0:
            message = f"[Invalid Tool Call] Start time must be greater than 0."
            message += remaining_turn_message
            return ToolResponse(text=message), 0.0, {}

        video_duration = self._instance_dict[instance_id]["video_duration"]
        if video_duration is not None and round(end_time, 1) > video_duration:
            message = f"[Invalid Tool Call] End time {end_time} exceeds video duration {video_duration}s."
            message += remaining_turn_message
            return ToolResponse(text=message), 0.0, {}

        pil_images, sample_fps = await self.extract_and_save_frames(
            instance_id=instance_id,
            video_start=start_time,
            video_end=end_time,
        )

        message = f"Successfully extracted {len(pil_images)} frames from {start_time}s to {end_time}s."
        message += remaining_turn_message
        vision_info = [sample_fps, start_time, end_time]

        return ToolResponse(text=message, image=pil_images, vision_info=vision_info), 0.05, {}

    # async def calc_reward(self, instance_id: str, **kwargs) -> float:
    #     return vstar.compute_score(
    #         self._instance_dict[instance_id]["response"],
    #         self._instance_dict[instance_id]["ground_truth"]
    #     )

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
