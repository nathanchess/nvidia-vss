######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
######################################################################################################

from typing import List, Tuple, Dict

import torch

from chunk_info import ChunkInfo
from via_logger import logger, TimeMeasure
from base_class import VlmGenerationConfig


class TwelveLabsContext:
    """Context management for unified Twelve Labs model."""
    
    def __init__(self, model):
        """Initialize Twelve Labs context.
        
        Args:
            model: The unified Twelve Labs model instance
        """
        self._model = model
        self._chunks = []
        self._video_embeds = []
        self._video_frames = []
        self._video_frames_times = []
        
    def set_video_embeds(
        self,
        chunks: List[ChunkInfo],
        video_embeds: List[torch.Tensor],
        video_frames: List[torch.Tensor],
        video_frames_times: List[List],
    ):
        """Set video embeddings and related data.
        
        Args:
            chunks: List of chunk information  
            video_embeds: List of video embeddings (not used for Twelve Labs)
            video_frames: List of video frames (not used for Twelve Labs)
            video_frames_times: List of frame timestamps
        """
        self._chunks = chunks
        self._video_embeds = video_embeds
        self._video_frames = video_frames
        self._video_frames_times = video_frames_times
    
    def prepare_inputs(
        self,
        chunks: List[ChunkInfo],
        frames: torch.Tensor,
        frame_times: List[float],
        prompts: List[str],
        **kwargs,
    ) -> dict:
        """Prepare inputs for Twelve Labs model.
        
        Args:
            chunks: List of chunk information
            frames: Video frames tensor
            frame_times: Frame timestamps
            prompts: List of prompts
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with prepared inputs
        """
        return {
            "chunks": chunks,
            "frames": frames,
            "frame_times": frame_times,
            "prompts": prompts,
            **kwargs
        }
    
    def parse_outputs(self, outputs: dict, **kwargs) -> List[str]:
        """Parse outputs from Twelve Labs model.
        
        Args:
            outputs: Model outputs
            **kwargs: Additional arguments
            
        Returns:
            List of parsed output strings
        """
        if isinstance(outputs, dict) and "text" in outputs:
            return [outputs["text"]]
        elif isinstance(outputs, list):
            return [str(output) for output in outputs]
        else:
            return [str(outputs)]
    
    def ask(self, prompt: str, **kwargs) -> Tuple[List[str], List[Dict]]:
        generation_config = kwargs.get("generation_config")
        chunk = kwargs.get("chunk", self._chunks)
        
        try:
            responses = []
            stats = []
            for chunk_info in chunk:
                response = self._process_chunk_unified(prompt, chunk_info, generation_config)
                responses.append(response)
                # Add placeholder stats for each chunk to match batching expectations
                stats.append({"status": "completed", "model": "twelve_labs"})
            
            return responses, stats
                
        except Exception as e:
            logger.error(f"Twelve Labs processing error: {e}")
            error_responses = [f"Error: {str(e)}"] * len(chunk)
            error_stats = [{"status": "error", "model": "twelve_labs"}] * len(chunk)
            return error_responses, error_stats
    
    def _process_chunk_unified(self, prompt: str, chunk_info: ChunkInfo, generation_config: VlmGenerationConfig) -> str:
        """Process a single chunk using the unified Twelve Labs workflow.
        
        Args:
            prompt: User prompt
            chunk_info: Chunk information
            generation_config: Generation configuration
            
        Returns:
            Response text from unified workflow
        """
        try:
            if generation_config is None:
                generation_config = VlmGenerationConfig()
            
            if isinstance(generation_config, dict):
                generation_config["prompt"] = prompt
            else:
                generation_config.prompt = prompt
            
            result = self._model.generate(
                chunks=[chunk_info],
                frames=None,
                frame_times=None,
                generation_config=generation_config,
            )
            
            if isinstance(result, dict):
                return result.get("text", "")
            else:
                return str(result)
                
        except Exception as e:
            logger.error(f"Error processing chunk with unified Twelve Labs: {e}")
            return f"Error in unified workflow: {str(e)}"