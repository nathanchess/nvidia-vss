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
import json
from typing import List, Optional, Dict, Generator, Any

import torch

from base_class import CustomModelBase, EmbeddingGeneratorBase, VlmGenerationConfig
from chunk_info import ChunkInfo
from via_logger import TimeMeasure, logger
from .twelve_labs_common import VideoIDMapper, wait_for_task_completion, ensure_index_exists, TwelveLabsConfig, retry_with_exponential_backoff

try:
    from twelvelabs import TwelveLabs
except ImportError:
    TwelveLabs = None
    logger.warning("TwelveLabs SDK not installed. Please install: pip install twelvelabs")


class TwelveLabsModel(CustomModelBase):
    def __init__(self, async_output: bool = True):
        self._client = None
        self._config = TwelveLabsConfig()
        self._initialize_client()
        
    def _initialize_client(self):
        if TwelveLabs is None:
            raise ImportError("TwelveLabs SDK not installed")
        
        if not self._config.validate():
            logger.error("Invalid Twelve Labs configuration")
            return
            
        try:
            self._client = TwelveLabs(api_key=self._config.api_key)
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
            stream = generation_config.get("stream", False)
        else:
            prompt = getattr(generation_config, "prompt", None) if generation_config else None
            stream = getattr(generation_config, "stream", False) if generation_config else False
        
        if not prompt:
            return {"text": "No prompt provided"}
        
        # Check if this is a search request with parameters
        analyze_mode = False
        max_clips = self._config.max_clips
        threshold = self._config.search_threshold
        
        # Parse search parameters from prompt if present
        if "[ANALYZE:" in prompt:
            import re
            # Extract parameters from prompt
            analyze_match = re.search(r'\[ANALYZE:\s*true.*?\]', prompt)
            if analyze_match:
                analyze_mode = True
                # Extract MAX_CLIPS parameter
                clips_match = re.search(r'MAX_CLIPS:\s*(\d+)', prompt)
                if clips_match:
                    max_clips = int(clips_match.group(1))
                # Extract THRESHOLD parameter
                threshold_match = re.search(r'THRESHOLD:\s*(\w+)', prompt)
                if threshold_match:
                    threshold = threshold_match.group(1)
                
                # Clean the prompt by removing the parameter section
                prompt = re.sub(r'\s*\[ANALYZE:.*?\]', '', prompt).strip()
        
        # Update config with search parameters
        if max_clips != self._config.max_clips:
            self._config.max_clips = max_clips
        if threshold != self._config.search_threshold:
            self._config.search_threshold = threshold
        
        if analyze_mode or not "[ANALYZE:" in str(generation_config):
            # Use search and analyze workflow
            if stream:
                return self._search_and_analyze_stream(prompt)
            else:
                result = self._search_and_analyze(prompt)
                return {"text": result["text"]}
        else:
            # For pure search (no analysis), return search results only
            result = self._search_only(prompt)
            return {"text": result["text"]}
    
    def _search_only(self, prompt: str) -> dict:
        """Execute search-only workflow: return clips without analysis."""
        try:
            logger.info(f"Starting search-only workflow")
            
            marengo_index_id = self._get_marengo_index_id()
            
            logger.info("Searching across all previously uploaded videos")
            relevant_clips = self._marengo_search_all(prompt, marengo_index_id)
            if not relevant_clips or len(relevant_clips) == 0:
                logger.warning("No relevant clips found")
                return {
                    "text": f"No relevant clips found for query: {prompt}",
                    "workflow": "marengo_search_no_results"
                }
            
            # Format search results without analysis
            search_results = f"Found {len(relevant_clips)} relevant clips for query: {prompt}\n\n"
            
            for i, clip in enumerate(relevant_clips[:10]):  # Show top 10 clips
                video_title = clip.get('video_title', f"Video {clip['video_id'][:8]}")
                search_results += f"{i+1}. {video_title}\n"
                search_results += f"   Time: {clip['start']:.1f}s - {clip['end']:.1f}s\n"
                search_results += f"   Score: {clip['score']:.2f}\n"
                search_results += f"   Confidence: {clip['confidence']}\n\n"
            
            return {
                "text": search_results,
                "clips_found": len(relevant_clips),
                "workflow": "marengo_search_only"
            }
            
        except Exception as e:
            logger.error(f"Twelve Labs search error: {e}")
            return {"text": f"Search error: {str(e)}", "error": True}
    
    def _search_and_analyze(self, prompt: str) -> dict:
        """Execute the search workflow: Marengo search → Pegasus analysis.
        
        Searches across ALL videos that have already been uploaded to Twelve Labs.
        Uses multiple clips for richer analysis.
        
        Args:
            prompt: User search prompt
            
        Returns:
            Final analysis result from matching clips across all videos
        """
        try:
            logger.info(f"Starting search workflow: Marengo search → Pegasus analysis")
            
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
            
            logger.info("Generating analysis from found clips")
            analysis = self._pegasus_analyze_clips(prompt, relevant_clips, pegasus_index_id)
            
            return {
                "text": analysis,
                "clips_found": len(relevant_clips),
                "workflow": "marengo_search_pegasus_analysis"
            }
            
        except Exception as e:
            logger.error(f"Twelve Labs search error: {e}")
            return {"text": f"Search error: {str(e)}", "error": True}
    
    def _search_and_analyze_stream(self, prompt: str) -> Generator[dict, None, None]:
        """Execute the search workflow with streaming response.
        
        Args:
            prompt: User search prompt
            
        Yields:
            Streaming response chunks in VSS format
        """
        try:
            logger.info(f"Starting streaming search workflow")
            
            marengo_index_id = self._get_marengo_index_id()
            pegasus_index_id = self._get_pegasus_index_id()
            
            # Yield initial status
            yield {"choices": [{"delta": {"content": "Searching across all videos...\n"}}]}
            
            relevant_clips = self._marengo_search_all(prompt, marengo_index_id)
            if not relevant_clips or len(relevant_clips) == 0:
                yield {"choices": [{"message": {"content": f"No relevant clips found for query: {prompt}"}}]}
                return
            
            yield {"choices": [{"delta": {"content": f"Found {len(relevant_clips)} relevant clips. Analyzing...\n"}}]}
            
            # Stream the analysis
            for chunk in self._pegasus_analyze_clips_stream(prompt, relevant_clips, pegasus_index_id):
                yield chunk
                
        except Exception as e:
            logger.error(f"Twelve Labs streaming search error: {e}")
            yield {"choices": [{"message": {"content": f"Search error: {str(e)}"}}]}
    
    def _get_marengo_index_id(self):
        def create_index():
            return ensure_index_exists(
                self._client,
                self._config.marengo_index_name,
                self._config.marengo_model,
                self._config.marengo_options
            )
        
        marengo_index_id = retry_with_exponential_backoff(
            create_index,
            self._config.max_retries,
            self._config.retry_delay_base
        )
        logger.info(f"Marengo index ready: {marengo_index_id}")
        return marengo_index_id
    
    def _get_pegasus_index_id(self):
        def create_index():
            return ensure_index_exists(
                self._client,
                self._config.pegasus_index_name,
                self._config.pegasus_model,
                self._config.pegasus_options
            )
        
        pegasus_index_id = retry_with_exponential_backoff(
            create_index,
            self._config.max_retries,
            self._config.retry_delay_base
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
            
            def perform_search():
                return self._client.search.query(
                    index_id=marengo_index_id,
                    options=self._config.search_options,
                    query_text=query,
                    group_by="clip",  # Get individual clips
                    threshold=self._config.search_threshold,
                    page_limit=self._config.max_clips,
                    sort_option="score"
                )
            
            search_results = retry_with_exponential_backoff(
                perform_search,
                self._config.max_retries,
                self._config.retry_delay_base
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
    
    def _pegasus_analyze_clips(self, prompt: str, clips: List[Dict], pegasus_index_id: str) -> str:
        try:
            logger.info(f"Pegasus analysis: {len(clips)} clips")
            
            # Use multiple clips for richer analysis
            top_clips = clips[:min(len(clips), self._config.max_clips_for_analysis)]
            
            # Build context from multiple clips
            clips_context = []
            pegasus_video_ids = set()
            
            for clip in top_clips:
                video_title = clip.get('video_title', f"Video {clip['video_id'][:8]}")
                pegasus_video_id = self._get_pegasus_video_id_for_marengo(clip['video_id'])
                if pegasus_video_id:
                    pegasus_video_ids.add(pegasus_video_id)
                    clips_context.append({
                        'video_id': pegasus_video_id,
                        'title': video_title,
                        'start': clip['start'],
                        'end': clip['end'],
                        'score': clip['score']
                    })
            
            if not clips_context:
                return "Could not generate analysis - video mapping not found"
            
            # Use the best clip's video for analysis, but include context from others
            primary_video_id = clips_context[0]['video_id']
            
            analysis_prompt = f"""Question: {prompt}

Relevant video segments found:
"""
            for i, clip in enumerate(clips_context[:3]):  # Show top 3 clips
                analysis_prompt += f"{i+1}. {clip['title']} ({clip['start']:.1f}s - {clip['end']:.1f}s) - Score: {clip['score']:.2f}\n"
            
            analysis_prompt += f"\nAnalyze the content and provide a comprehensive answer to the question."
            
            # Use analyze instead of summarize for custom prompts
            response = self._client.analyze(
                video_id=primary_video_id,
                prompt=analysis_prompt,
                temperature=self._config.analysis_temperature
            )
            
            if hasattr(response, 'text') and response.text:
                analysis_text = response.text
                logger.info(f"Analysis generated: {len(analysis_text)} chars")
            else:
                logger.warning(f"No analysis in response: {type(response)}")
                video_count = len(set(clip['video_id'] for clip in clips))
                analysis_text = f"Found {len(clips)} relevant clips from {video_count} videos for query: {prompt}"
            
            logger.info("Pegasus analysis completed")
            return analysis_text
            
        except Exception as e:
            logger.error(f"Pegasus analysis error: {e}")
            video_count = len(set(clip['video_id'] for clip in clips))
            return f"Found {len(clips)} relevant clips from {video_count} videos for query: {prompt}"
    
    def _pegasus_analyze_clips_stream(self, prompt: str, clips: List[Dict], pegasus_index_id: str) -> Generator[dict, None, None]:
        """Stream analysis results from Pegasus."""
        try:
            logger.info(f"Pegasus streaming analysis: {len(clips)} clips")
            
            # Use multiple clips for richer analysis
            top_clips = clips[:min(len(clips), self._config.max_clips_for_analysis)]
            
            # Build context from multiple clips
            clips_context = []
            for clip in top_clips:
                video_title = clip.get('video_title', f"Video {clip['video_id'][:8]}")
                pegasus_video_id = self._get_pegasus_video_id_for_marengo(clip['video_id'])
                if pegasus_video_id:
                    clips_context.append({
                        'video_id': pegasus_video_id,
                        'title': video_title,
                        'start': clip['start'],
                        'end': clip['end'],
                        'score': clip['score']
                    })
            
            if not clips_context:
                yield {"choices": [{"message": {"content": "Could not generate analysis - video mapping not found"}}]}
                return
            
            # Use the best clip's video for analysis
            primary_video_id = clips_context[0]['video_id']
            
            analysis_prompt = f"""Question: {prompt}

Relevant video segments found:
"""
            for i, clip in enumerate(clips_context[:3]):  # Show top 3 clips
                analysis_prompt += f"{i+1}. {clip['title']} ({clip['start']:.1f}s - {clip['end']:.1f}s) - Score: {clip['score']:.2f}\n"
            
            analysis_prompt += f"\nAnalyze the content and provide a comprehensive answer to the question."
            
            # Use streaming analyze
            try:
                for chunk in self._client.analyze_stream(
                    video_id=primary_video_id,
                    prompt=analysis_prompt,
                    temperature=self._config.analysis_temperature
                ):
                    if hasattr(chunk, 'text') and chunk.text:
                        yield {"choices": [{"delta": {"content": chunk.text}}]}
                    elif hasattr(chunk, 'content') and chunk.content:
                        yield {"choices": [{"delta": {"content": chunk.content}}]}
                        
                # Send completion marker
                yield {"choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]}
                
            except Exception as stream_error:
                logger.warning(f"Streaming failed, falling back to regular analysis: {stream_error}")
                # Fallback to non-streaming
                response = self._client.analyze(
                    video_id=primary_video_id,
                    prompt=analysis_prompt,
                    temperature=self._config.analysis_temperature
                )
                
                if hasattr(response, 'text') and response.text:
                    yield {"choices": [{"message": {"content": response.text}}]}
                else:
                    video_count = len(set(clip['video_id'] for clip in clips))
                    yield {"choices": [{"message": {"content": f"Found {len(clips)} relevant clips from {video_count} videos for query: {prompt}"}}]}
            
        except Exception as e:
            logger.error(f"Pegasus streaming analysis error: {e}")
            video_count = len(set(clip['video_id'] for clip in clips))
            yield {"choices": [{"message": {"content": f"Found {len(clips)} relevant clips from {video_count} videos for query: {prompt}"}}]}
    
    def _get_pegasus_video_id_for_marengo(self, marengo_video_id: str) -> Optional[str]:
        """Get the corresponding Pegasus video ID for a Marengo video ID."""
        # Use optimized lookup from VideoIDMapper
        return VideoIDMapper.get_pegasus_for_marengo(marengo_video_id)
    
    def get_embedding_generator(self):
        return None
    
    @staticmethod
    def get_model_info():
        return "twelve-labs", "cloud", "twelve-labs"
    
    def warmup(self):
        pass
