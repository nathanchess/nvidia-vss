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

"""
Search Handler for Twelve Labs cross-video search functionality.
Separate from video processing pipeline - dedicated to search operations.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Dict, List, Optional, Generator, Any

from via_logger import logger
from via_exception import ViaException


class SearchRequestInfo:
    """Store information for a search request"""
    
    class Status(Enum):
        """Search Request Status."""
        QUEUED = "queued"
        PROCESSING = "processing"
        SUCCESSFUL = "successful"
        FAILED = "failed"
        STOPPING = "stopping"
    
    class Response:
        def __init__(self, response: str, timestamp: str = None) -> None:
            self.response = response
            self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
    
    def __init__(self) -> None:
        self.request_id = str(uuid.uuid4())
        self.status = SearchRequestInfo.Status.QUEUED
        self.query = ""
        self.model = ""
        self.analyze = False
        self.stream = False
        self.max_clips = 10
        self.threshold = "medium"
        self.temperature = 0.7
        self.queue_time = time.time()
        self.start_time = None
        self.end_time = None
        self.processing_time = 0
        self.responses: List[SearchRequestInfo.Response] = []
        self.error_message = ""


class SearchHandler:
    """Handler for cross-video search operations using Twelve Labs."""
    
    def __init__(self):
        self._requests: Dict[str, SearchRequestInfo] = {}
        self._requests_lock = RLock()
        self._twelve_labs_model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Twelve Labs model."""
        try:
            # Import and initialize the Twelve Labs model
            from models.twelve_labs.twelve_labs_model import TwelveLabsModel
            self._twelve_labs_model = TwelveLabsModel()
            logger.info("SearchHandler: Twelve Labs model initialized")
        except Exception as e:
            logger.error(f"SearchHandler: Failed to initialize Twelve Labs model: {e}")
            self._twelve_labs_model = None
    
    def search(
        self,
        query: str,
        model: str = "twelve-labs",
        analyze: bool = False,
        stream: bool = False,
        max_clips: int = 10,
        threshold: str = "medium",
        temperature: float = 0.7
    ) -> str:
        """
        Execute a cross-video search request.
        
        Args:
            query: Search query text
            model: Model to use (should be "twelve-labs")
            analyze: Whether to analyze found clips or just return search results
            stream: Whether to stream responses
            max_clips: Maximum number of clips to return
            threshold: Search threshold (low, medium, high)
            temperature: Generation temperature for analysis
            
        Returns:
            Request ID for tracking the search
        """
        if not self._twelve_labs_model:
            raise ViaException("Twelve Labs model not available", "ModelError", 503)
        
        # Create search request info
        search_info = SearchRequestInfo()
        search_info.query = query
        search_info.model = model
        search_info.analyze = analyze
        search_info.stream = stream
        search_info.max_clips = max_clips
        search_info.threshold = threshold
        search_info.temperature = temperature
        search_info.start_time = time.time()
        
        # Store request
        with self._requests_lock:
            self._requests[search_info.request_id] = search_info
        
        logger.info(f"SearchHandler: Starting search request {search_info.request_id} - query: '{query}', analyze: {analyze}")
        
        try:
            # Update status to processing
            search_info.status = SearchRequestInfo.Status.PROCESSING
            
            # Prepare search parameters
            search_prompt = query
            if analyze:
                search_prompt += f" [ANALYZE: true, MAX_CLIPS: {max_clips}, THRESHOLD: {threshold}]"
            
            # Create generation config
            generation_config = {
                "prompt": search_prompt,
                "stream": stream,
                "temperature": temperature
            }
            
            # Execute search directly through Twelve Labs search methods
            if stream:
                # For streaming, we'll handle it differently
                search_info.status = SearchRequestInfo.Status.SUCCESSFUL
                logger.info(f"SearchHandler: Search request {search_info.request_id} queued for streaming")
            else:
                # Non-streaming search - use the dedicated search method
                result = self._twelve_labs_model.search(
                    query=query,
                    analyze=analyze,
                    stream=False,
                    max_clips=max_clips,
                    threshold=threshold
                )
                
                # Store response
                response_text = result.get("text", "No results found")
                search_info.responses.append(SearchRequestInfo.Response(response_text))
                search_info.status = SearchRequestInfo.Status.SUCCESSFUL
                search_info.end_time = time.time()
                search_info.processing_time = search_info.end_time - search_info.start_time
                
                logger.info(f"SearchHandler: Search request {search_info.request_id} completed in {search_info.processing_time:.2f}s")
        
        except Exception as e:
            logger.error(f"SearchHandler: Search request {search_info.request_id} failed: {e}")
            search_info.status = SearchRequestInfo.Status.FAILED
            search_info.error_message = str(e)
            search_info.end_time = time.time()
            search_info.processing_time = search_info.end_time - search_info.start_time
        
        return search_info.request_id
    
    def get_search_response(
        self, 
        request_id: str, 
        response_count: int = -1
    ) -> tuple[SearchRequestInfo, List[SearchRequestInfo.Response]]:
        """
        Get search response for a given request ID.
        
        Args:
            request_id: The search request ID
            response_count: Number of responses to return (-1 for all)
            
        Returns:
            Tuple of (SearchRequestInfo, List of responses)
        """
        with self._requests_lock:
            search_info = self._requests.get(request_id)
            if not search_info:
                return None, []
            
            if response_count == -1:
                responses = search_info.responses.copy()
            else:
                responses = search_info.responses[:response_count]
            
            return search_info, responses
    
    def get_search_stream_response(self, request_id: str) -> Generator[Dict[str, Any], None, None]:
        """
        Get streaming search response for a given request ID.
        
        Args:
            request_id: The search request ID
            
        Yields:
            Dictionary responses in streaming format
        """
        search_info = None
        with self._requests_lock:
            search_info = self._requests.get(request_id)
        
        if not search_info:
            logger.error(f"SearchHandler: Request {request_id} not found for streaming")
            return
        
        if not search_info.stream:
            logger.error(f"SearchHandler: Request {request_id} was not configured for streaming")
            return
        
        try:
            logger.info(f"SearchHandler: Starting streaming search {request_id}")
            search_info.status = SearchRequestInfo.Status.PROCESSING
            
            # Prepare search parameters
            search_prompt = search_info.query
            if search_info.analyze:
                search_prompt += f" [ANALYZE: true, MAX_CLIPS: 10, THRESHOLD: medium]"
            
            # Use the dedicated search method with streaming
            stream_generator = self._twelve_labs_model.search(
                query=search_info.query,
                analyze=search_info.analyze, 
                stream=True,
                max_clips=search_info.max_clips,
                threshold=search_info.threshold
            )
            for chunk in stream_generator:
                yield chunk
            
            search_info.status = SearchRequestInfo.Status.SUCCESSFUL
            search_info.end_time = time.time()
            search_info.processing_time = search_info.end_time - search_info.start_time
            
            logger.info(f"SearchHandler: Streaming search {request_id} completed")
            
        except Exception as e:
            logger.error(f"SearchHandler: Streaming search {request_id} failed: {e}")
            search_info.status = SearchRequestInfo.Status.FAILED
            search_info.error_message = str(e)
            yield {
                "choices": [{
                    "delta": {"content": f"Search error: {str(e)}"},
                    "index": 0
                }]
            }
    
    def check_status_remove_req_id(self, request_id: str) -> None:
        """Remove completed request from tracking."""
        with self._requests_lock:
            if request_id in self._requests:
                del self._requests[request_id]
                logger.debug(f"SearchHandler: Removed completed request {request_id}")
    
    async def wait_for_search_done(self, request_id: str, timeout: float = 300.0) -> bool:
        """
        Wait for search request to complete.
        
        Args:
            request_id: The search request ID
            timeout: Timeout in seconds
            
        Returns:
            True if completed successfully, False if timeout or failed
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._requests_lock:
                search_info = self._requests.get(request_id)
                if not search_info:
                    return False
                
                if search_info.status in [SearchRequestInfo.Status.SUCCESSFUL, SearchRequestInfo.Status.FAILED]:
                    return search_info.status == SearchRequestInfo.Status.SUCCESSFUL
            
            await asyncio.sleep(0.1)
        
        # Timeout
        logger.warning(f"SearchHandler: Search request {request_id} timed out after {timeout}s")
        return False