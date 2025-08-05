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
        """Generate video summarization using Twelve Labs Pegasus model ONLY."""
        logger.info(f"TwelveLabsModel.generate() called with {len(chunks) if chunks else 0} chunks")
        
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
        
        # This method is ONLY for summarization - NEVER do search
        if not chunks or len(chunks) == 0:
            logger.error("generate() method called without chunks - this should ONLY be used for summarization")
            return {"text": "ERROR: generate() method requires video chunks for summarization. Use search() method for search queries.", "error": True}
        
        logger.info(f"Starting Pegasus video summarization for {len(chunks)} chunks")
        if stream:
            return self._summarize_video_stream(prompt, chunks)
        else:
            result = self._summarize_video(prompt, chunks)
            return {"text": result["text"]}
    
    def _summarize_video(self, prompt: str, chunks: List[ChunkInfo]) -> dict:
        """Execute video summarization using Twelve Labs Pegasus model."""
        try:
            logger.info(f"Starting video summarization for {len(chunks)} chunks")
            
            # Get the video file path from the first chunk
            if not chunks or len(chunks) == 0:
                return {"text": "No video chunks provided for summarization", "error": True}
            
            # Get the VSS file ID from the chunk info
            chunk = chunks[0]
            video_file_path = chunk.file if hasattr(chunk, 'file') else None
            
            if not video_file_path:
                return {"text": "Unable to determine video file path from chunks", "error": True}
            
            # Extract VSS file ID from the path (typically in format: ./assets/{vss_id}/video.mp4)
            import re
            path_parts = str(video_file_path).split('/')
            vss_file_id = None
            for part in path_parts:
                if re.match(r'^[0-9a-f-]{36}$', part):  # UUID format
                    vss_file_id = part
                    break
            
            if not vss_file_id:
                return {"text": f"Unable to extract VSS file ID from path: {video_file_path}", "error": True}
            
            logger.info(f"Summarizing video with VSS ID: {vss_file_id}")
            
            # Ensure the video is uploaded to Twelve Labs
            upload_result = self.ensure_video_uploaded(vss_file_id)
            if upload_result.get("error"):
                return upload_result
            
            pegasus_video_id = upload_result.get("pegasus_video_id")
            if not pegasus_video_id:
                return {"text": "Failed to get Pegasus video ID for summarization", "error": True}
            
            # Use Twelve Labs Pegasus for actual video summarization
            pegasus_index_id = self._get_pegasus_index_id()
            
            logger.info(f"Using Pegasus to summarize video {pegasus_video_id}")
            response = self._client.summarize(
                video_id=pegasus_video_id,
                prompt=prompt,
                temperature=self._config.analysis_temperature,
                type="summary"
            )
            
            logger.info(f"Result ID: {response.id}")
            
            if response.summary is not None:
                summary_text = response.summary
                logger.info(f"Video summarization completed: {len(summary_text)} chars")
            else:
                logger.warning(f"No summary in response: {type(response)}")
                summary_text = f"Unable to generate summary for video {vss_file_id}"
            
            return {
                "text": summary_text,
                "vss_file_id": vss_file_id,
                "pegasus_video_id": pegasus_video_id,
                "workflow": "pegasus_summarization"
            }
            
        except Exception as e:
            logger.error(f"Video summarization error: {e}")
            return {"text": f"Summarization error: {str(e)}", "error": True}
    
    def _summarize_video_stream(self, prompt: str, chunks: List[ChunkInfo]) -> dict:
        """Execute streaming video summarization using Twelve Labs Pegasus model."""
        try:
            logger.info(f"Starting streaming video summarization for {len(chunks)} chunks")
            
            # Get the video file info same as non-streaming
            if not chunks or len(chunks) == 0:
                yield {"choices": [{"message": {"content": "No video chunks provided for summarization"}}]}
                return
            
            chunk = chunks[0]
            video_file_path = chunk.file if hasattr(chunk, 'file') else None
            
            if not video_file_path:
                yield {"choices": [{"message": {"content": "Unable to determine video file path from chunks"}}]}
                return
            
            # Extract VSS file ID from path
            import re
            path_parts = str(video_file_path).split('/')
            vss_file_id = None
            for part in path_parts:
                if re.match(r'^[0-9a-f-]{36}$', part):  # UUID format
                    vss_file_id = part
                    break
            
            if not vss_file_id:
                yield {"choices": [{"message": {"content": f"Unable to extract VSS file ID from path: {video_file_path}"}}]}
                return
            
            logger.info(f"Streaming summarization for video with VSS ID: {vss_file_id}")
            
            # Ensure video is uploaded
            upload_result = self.ensure_video_uploaded(vss_file_id)
            if upload_result.get("error"):
                yield {"choices": [{"message": {"content": upload_result["text"]}}]}
                return
            
            pegasus_video_id = upload_result.get("pegasus_video_id")
            if not pegasus_video_id:
                yield {"choices": [{"message": {"content": "Failed to get Pegasus video ID for summarization"}}]}
                return
            
            # Use Pegasus summarization (Twelve Labs may not have streaming summarize)
            logger.info(f"Streaming Pegasus summarization for video {pegasus_video_id}")
            try:
                # Try streaming summarize if available
                if hasattr(self._client, 'summarize_stream'):
                    for chunk in self._client.summarize_stream(
                        video_id=pegasus_video_id,
                        prompt=prompt,
                        temperature=self._config.analysis_temperature,
                        type="summary"
                    ):
                        if hasattr(chunk, 'summary') and chunk.summary:
                            yield {"choices": [{"delta": {"content": chunk.summary}}]}
                        elif hasattr(chunk, 'text') and chunk.text:
                            yield {"choices": [{"delta": {"content": chunk.text}}]}
                else:
                    # Fallback to non-streaming summarize
                    response = self._client.summarize(
                        video_id=pegasus_video_id,
                        prompt=prompt,
                        temperature=self._config.analysis_temperature,
                        type="summary"
                    )
                    
                    if response.summary is not None:
                        yield {"choices": [{"message": {"content": response.summary}}]}
                    else:
                        yield {"choices": [{"message": {"content": f"Unable to generate summary for video {vss_file_id}"}}]}
                
                # Send completion marker
                yield {"choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]}
                
            except Exception as stream_error:
                logger.warning(f"Streaming failed, falling back to regular summarize: {stream_error}")
                # Fallback to non-streaming
                response = self._client.summarize(
                    video_id=pegasus_video_id,
                    prompt=prompt,
                    temperature=self._config.analysis_temperature,
                    type="summary"
                )
                
                if response.summary is not None:
                    yield {"choices": [{"message": {"content": response.summary}}]}
                else:
                    yield {"choices": [{"message": {"content": f"Unable to generate summary for video {vss_file_id}"}}]}
                    
        except Exception as e:
            logger.error(f"Streaming video summarization error: {e}")
            yield {"choices": [{"message": {"content": f"Summarization error: {str(e)}"}}]}
    
    def search(self, query: str, analyze: bool = False, stream: bool = False, max_clips: int = None, threshold: str = None):
        """Execute search across all videos using Twelve Labs Marengo model."""
        if self._client is None:
            return {"text": "Client not available"}
        
        logger.info(f"Starting Twelve Labs search: query='{query}', analyze={analyze}, stream={stream}")
        
        # Set search parameters
        if max_clips is not None:
            self._config.max_clips = max_clips
        if threshold is not None:
            self._config.search_threshold = threshold
        
        if stream:
            return self._search_stream(query)
        else:
            result = self._search_only(query)
            return {"text": result["text"]}
    
    def _search_stream(self, query: str):
        """Execute streaming search across all videos."""
        for chunk in self._search_only_stream(query):
            yield chunk
    
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
                vss_file_id = clip.get('vss_file_id', 'Unknown')
                twelve_labs_video_id = clip.get('video_id', 'Unknown')
                search_results += f"{i+1}. {video_title}\n"
                search_results += f"   VSS Video ID: {vss_file_id}\n"
                search_results += f"   Twelve Labs Video ID: {twelve_labs_video_id}\n"
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
                    logger.info(f"Processing search result: video_id={result.video_id}")
                    vss_file_id = VideoIDMapper.get_vss_id_for_marengo(result.video_id)
                    logger.info(f"Found VSS file ID: {vss_file_id} for Marengo video ID: {result.video_id}")
                    clips.append({
                        "start": result.start,
                        "end": result.end,
                        "score": result.score,
                        "confidence": result.confidence,
                        "video_id": result.video_id,
                        "video_title": getattr(result, 'video_title', 'Unknown'),
                        "vss_file_id": vss_file_id or 'Unknown'
                    })
            else:
                for page in search_results:
                    for result in page:
                        logger.info(f"Processing search result: video_id={result.video_id}")
                        vss_file_id = VideoIDMapper.get_vss_id_for_marengo(result.video_id)
                        logger.info(f"Found VSS file ID: {vss_file_id} for Marengo video ID: {result.video_id}")
                        clips.append({
                            "start": result.start,
                            "end": result.end,
                            "score": result.score,
                            "confidence": result.confidence,
                            "video_id": result.video_id,
                            "video_title": getattr(result, 'video_title', 'Unknown'),
                            "vss_file_id": vss_file_id or 'Unknown'
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
                else:
                    logger.warning(f"Could not find Pegasus video ID for Marengo video ID {clip['video_id']}")
            
            if not clips_context:
                error_msg = f"Could not generate analysis - no video mappings found for {len(top_clips)} clips"
                logger.error(error_msg)
                return error_msg
            
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
                else:
                    logger.warning(f"Could not find Pegasus video ID for Marengo video ID {clip['video_id']}")
            
            if not clips_context:
                error_msg = f"Could not generate analysis - no video mappings found for {len(top_clips)} clips"
                logger.error(error_msg)
                yield {"choices": [{"message": {"content": error_msg}}]}
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
