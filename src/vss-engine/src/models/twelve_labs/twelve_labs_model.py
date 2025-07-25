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

import os
import time
from typing import List, Optional, Dict

import torch

from base_class import CustomModelBase, EmbeddingGeneratorBase, VlmGenerationConfig
from chunk_info import ChunkInfo
from via_logger import TimeMeasure, logger
from .twelve_labs_common import VideoIDMapper, wait_for_task_completion, ensure_index_exists

try:
    from twelvelabs import TwelveLabs
except ImportError:
    TwelveLabs = None
    logger.warning("TwelveLabs SDK not installed. Please install: pip install twelvelabs")


class TwelveLabsModel(CustomModelBase):
    def __init__(self, async_output: bool = True):
        self._client = None
        self._initialize_client()
        
    def _initialize_client(self):
        if TwelveLabs is None:
            raise ImportError("TwelveLabs SDK not installed")
        
        api_key = os.environ.get("TWELVE_LABS_API_KEY")
        if not api_key:
            logger.warning("TWELVE_LABS_API_KEY not set")
            return
            
        try:
            self._client = TwelveLabs(api_key=api_key)
            indexes = list(self._client.index.list())
            logger.info(f"Connected to Twelve Labs. Found {len(indexes)} indexes")
        except Exception as e:
            logger.error(f"Failed to initialize client: {e}")
            self._client = None
    
    def generate(self, chunks: List[ChunkInfo], frames: torch.Tensor, frame_times: List[float], 
                 generation_config: VlmGenerationConfig = None, **kwargs) -> dict:
        if self._client is None:
            return {"text": "Client not available"}
        
        if isinstance(generation_config, dict):
            prompt = generation_config.get("prompt")
        else:
            prompt = getattr(generation_config, "prompt", None) if generation_config else None
        
        if not prompt:
            return {"text": "No prompt provided"}
        
        result = self._search_and_summarize(prompt)
        return {"text": result["text"]}
    
    def _search_and_summarize(self, prompt: str) -> dict:
        """Execute the search workflow: Marengo search → Pegasus summary.
        
        Searches across ALL videos that have already been uploaded to Twelve Labs.
        Videos should be uploaded during VSS upload, not during search.
        
        Args:
            prompt: User search prompt
            
        Returns:
            Final summary result from matching clips across all videos
        """
        try:
            logger.info(f"Starting search workflow: Marengo search → Pegasus summary")
            
            marengo_index_id = self._get_marengo_index_id()
            pegasus_index_id = self._get_pegasus_index_id()
            
            logger.info("Searching across all previously uploaded videos")
            relevant_clips = self._marengo_search_all(prompt, marengo_index_id)
            if not relevant_clips or len(relevant_clips) == 0:
                logger.warning("No relevant clips found")
                return {
                    "text": f"No relevant clips found for query: {prompt}",
                    "workflow": "marengo_search_no_results"
                }
            
            logger.info("Generating summary from found clips")
            summary = self._pegasus_summarize_clips(prompt, relevant_clips, pegasus_index_id)
            
            return {
                "text": summary,
                "clips_found": len(relevant_clips),
                "workflow": "marengo_search_pegasus_summary"
            }
            
        except Exception as e:
            logger.error(f"Twelve Labs search error: {e}")
            return {"text": f"Search error: {str(e)}", "error": True}
    
    def _get_marengo_index_id(self):
        marengo_index_name = os.environ.get("TWELVE_LABS_MARENGO_INDEX_NAME")
        marengo_model_name = os.environ.get("TWELVE_LABS_MARENGO_MODEL")
        marengo_options = os.environ.get("TWELVE_LABS_MARENGO_OPTIONS").split(",")
        marengo_index_id = ensure_index_exists(
            self._client,
            marengo_index_name,
            marengo_model_name,
            marengo_options
        )
        logger.info(f"Marengo index ready: {marengo_index_id}")
        return marengo_index_id
    
    def _get_pegasus_index_id(self):
        pegasus_index_name = os.environ.get("TWELVE_LABS_PEGASUS_INDEX_NAME")
        pegasus_model_name = os.environ.get("TWELVE_LABS_PEGASUS_MODEL")
        pegasus_options = os.environ.get("TWELVE_LABS_PEGASUS_OPTIONS").split(",")
        pegasus_index_id = ensure_index_exists(
            self._client,
            pegasus_index_name,
            pegasus_model_name,
            pegasus_options
        )
        logger.info(f"Pegasus index ready: {pegasus_index_id}")
        return pegasus_index_id
    
    def ensure_video_uploaded(self, vss_file_id: str) -> Dict:
        try:
            marengo_index_id = self._get_marengo_index_id()
            pegasus_index_id = self._get_pegasus_index_id()
        except Exception as e:
            logger.error(f"Failed to get indexes: {e}")
            return {"text": f"Failed to initialize indexes: {str(e)}", "error": True}
        
        mapping = VideoIDMapper.get_mapping(vss_file_id)
        marengo_video_id = mapping.get("marengo_video_id") if mapping else None
        pegasus_video_id = mapping.get("pegasus_video_id") if mapping else None
        
        video_path = VideoIDMapper.get_video_path(vss_file_id)
        if not video_path:
            return {"text": f"Video file not found for ID: {vss_file_id}", "error": True}
        
        if not marengo_video_id:
            logger.info(f"Uploading video to Marengo index")
            marengo_result = self._upload_video_to_index(video_path, marengo_index_id, "marengo")
            if marengo_result.get("error"):
                return marengo_result
            marengo_video_id = marengo_result.get("video_id")
            VideoIDMapper.save_mapping(vss_file_id, marengo_video_id=marengo_video_id, marengo_index_id=marengo_index_id)
        
        if not pegasus_video_id:
            logger.info(f"Uploading video to Pegasus index")
            pegasus_result = self._upload_video_to_index(video_path, pegasus_index_id, "pegasus")
            if pegasus_result.get("error"):
                return pegasus_result
            pegasus_video_id = pegasus_result.get("video_id")
            VideoIDMapper.save_mapping(vss_file_id, pegasus_video_id=pegasus_video_id, pegasus_index_id=pegasus_index_id)
        
        return {
            "marengo_video_id": marengo_video_id,
            "pegasus_video_id": pegasus_video_id
        }
    
    def _upload_video_to_index(self, video_path, index_id: str, model_name: str) -> Dict:
        try:
            logger.info(f"Uploading to {model_name}: video_path={video_path}, index_id={index_id}")
            logger.info(f"File exists: {video_path.exists()}, File size: {video_path.stat().st_size if video_path.exists() else 'N/A'}")
            logger.info(f"Client initialized: {self._client is not None}")
            
            if not video_path.exists():
                return {"text": f"Video file not found: {video_path}", "error": True}
            
            if not index_id:
                return {"text": f"No index ID provided for {model_name}", "error": True}
            
            if not self._client:
                return {"text": f"Twelve Labs client not initialized", "error": True}
            
            logger.info(f"Creating upload task for {model_name} with index {index_id}")
            task = self._client.task.create(
                index_id=index_id,
                file=str(video_path)
            )
            
            logger.info(f"Waiting for {model_name} upload task {task.id} to complete...")
            task_result = wait_for_task_completion(self._client, task.id)
            
            if task_result.get("status") != "ready":
                return {"text": f"{model_name} upload failed: {task_result.get('error', 'Unknown error')}", "error": True}
            
            video_id = task_result.get("video_id")
            logger.info(f"{model_name} upload successful, video ID: {video_id}")
            return {"video_id": video_id}
            
        except Exception as e:
            logger.error(f"Error uploading to {model_name}: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            return {"text": f"Video upload failed: {str(e)}", "error": True}
    
    def _marengo_search_all(self, query: str, marengo_index_id: str) -> List[Dict]:
        try:
            logger.info(f"Marengo search across all videos: '{query}'")
            
            search_options = os.environ.get("TWELVE_LABS_SEARCH_OPTIONS").split(",")
            max_clips = int(os.environ.get("TWELVE_LABS_MAX_CLIPS"))
            
            search_results = self._client.search.query(
                index_id=marengo_index_id,
                options=search_options,  # ["visual", "audio"]
                query_text=query,
                group_by="clip",  # Get individual clips
                threshold="medium",  # Medium confidence threshold
                page_limit=max_clips,
                sort_option="score"
            )
            
            clips = []
            if hasattr(search_results, 'data'):
                for result in search_results.data:
                    clips.append({
                        "start": result.start,
                        "end": result.end,
                        "score": result.score,
                        "confidence": result.confidence,
                        "video_id": result.video_id,
                        "video_title": getattr(result, 'video_title', 'Unknown')
                    })
            else:
                for page in search_results:
                    for result in page:
                        clips.append({
                            "start": result.start,
                            "end": result.end,
                            "score": result.score,
                            "confidence": result.confidence,
                            "video_id": result.video_id,
                            "video_title": getattr(result, 'video_title', 'Unknown')
                        })
            
            logger.info(f"Marengo found {len(clips)} relevant clips across all videos")
            return clips
            
        except Exception as e:
            logger.error(f"Marengo search error: {e}")
            return []
    
    def _pegasus_summarize_clips(self, prompt: str, clips: List[Dict], pegasus_index_id: str) -> str:
        try:
            logger.info(f"Pegasus summarization: {len(clips)} clips")
            
            best_clip = clips[0]
            video_title = best_clip.get('video_title', f"Video {best_clip['video_id'][:8]}")
            
            summary_prompt = f"""Question: {prompt}

Analyze this video segment: {video_title} ({best_clip['start']:.1f}s - {best_clip['end']:.1f}s)

Provide a summary that answers the question."""
            
            best_marengo_video_id = clips[0]['video_id'] if clips else None
            if not best_marengo_video_id:
                return "No clips found"
            
            pegasus_video_id = None
            
            import os
            from pathlib import Path
            asset_storage_dir = os.environ.get("ASSET_STORAGE_DIR")
            
            for asset_dir in Path(asset_storage_dir).iterdir():
                if asset_dir.is_dir():
                    mapping_file = asset_dir / "twelve_labs_mapping.json"
                    if mapping_file.exists():
                        mapping = VideoIDMapper.get_mapping(asset_dir.name)
                        if mapping and mapping.get("marengo_video_id") == best_marengo_video_id:
                            pegasus_video_id = mapping.get("pegasus_video_id")
                            logger.info(f"Found Pegasus video ID {pegasus_video_id} for Marengo video ID {best_marengo_video_id}")
                            break
            
            if not pegasus_video_id:
                logger.error(f"No Pegasus video ID found for Marengo video {best_marengo_video_id}")
                return "Could not generate summary - video mapping not found"
            
            response = self._client.summarize(
                video_id=pegasus_video_id,
                type="summary",  # Generate a summary
                prompt=summary_prompt,
                temperature=0.7
            )
            
            if hasattr(response, 'summary') and response.summary:
                summary_text = response.summary
                logger.info(f"Summary generated: {len(summary_text)} chars")
            else:
                logger.warning(f"No summary in response: {type(response)}")
                summary_text = "Summary generation failed"
            
            logger.info("Pegasus summarization completed")
            return summary_text
            
        except Exception as e:
            logger.error(f"Pegasus summarization error: {e}")
            video_count = len(set(clip['video_id'] for clip in clips))
            return f"Found {len(clips)} relevant clips from {video_count} videos for query: {prompt}"
    
    def get_embedding_generator(self):
        return None
    
    @staticmethod
    def get_model_info():
        return "twelve-labs", "cloud", "twelve-labs"
    
    def warmup(self):
        pass
