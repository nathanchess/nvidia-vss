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

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

from via_logger import logger

ASSET_STORAGE_DIR = os.environ.get("ASSET_STORAGE_DIR")


class VideoIDMapper:
    @staticmethod
    def save_mapping(vss_file_id: str, marengo_video_id: str = None, 
                     pegasus_video_id: str = None, marengo_index_id: str = None,
                     pegasus_index_id: str = None) -> None:
        asset_dir = Path(ASSET_STORAGE_DIR) / vss_file_id
        if not asset_dir.exists():
            asset_dir = Path(ASSET_STORAGE_DIR) / vss_file_id.replace("-", "")
        
        mapping_file = asset_dir / "twelve_labs_mapping.json"
        
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                mapping = json.load(f)
        else:
            mapping = {"vss_file_id": vss_file_id}
        
        if marengo_video_id:
            mapping["marengo_video_id"] = marengo_video_id
        if pegasus_video_id:
            mapping["pegasus_video_id"] = pegasus_video_id
        if marengo_index_id:
            mapping["marengo_index_id"] = marengo_index_id
        if pegasus_index_id:
            mapping["pegasus_index_id"] = pegasus_index_id
        
        with open(mapping_file, 'w') as f:
            json.dump(mapping, f)
    
    @staticmethod
    def get_mapping(vss_file_id: str) -> Optional[Dict]:
        for file_id in [vss_file_id, vss_file_id.replace("-", "")]:
            mapping_file = Path(ASSET_STORAGE_DIR) / file_id / "twelve_labs_mapping.json"
            if mapping_file.exists():
                with open(mapping_file, 'r') as f:
                    return json.load(f)
        return None
    
    @staticmethod
    def get_video_path(vss_file_id: str) -> Optional[Path]:
        for file_id in [vss_file_id, vss_file_id.replace("-", "")]:
            asset_dir = Path(ASSET_STORAGE_DIR) / file_id
            if asset_dir.exists():
                for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                    video_files = list(asset_dir.glob(f"*{ext}"))
                    if video_files:
                        return video_files[0]
        return None


def wait_for_task_completion(client, task_id: str) -> Dict:
    max_wait_seconds = int(os.environ.get("TWELVE_LABS_UPLOAD_TIMEOUT"))
    start_time = time.time()
    
    while time.time() - start_time < max_wait_seconds:
        try:
            task = client.task.retrieve(task_id)
            
            if task.status == "ready":
                return {"status": "ready", "video_id": task.video_id}
            elif task.status == "failed":
                return {"status": "failed", "error": "Task failed"}
            
            time.sleep(5)
            
        except Exception as e:
            logger.error(f"Error checking task: {e}")
            time.sleep(5)
    
    return {"status": "timeout", "error": "Upload timeout"}


def ensure_index_exists(client, index_name: str, engine_name: str, engine_options: list) -> str:
    try:
        indexes = list(client.index.list())
        
        for index in indexes:
            if index.name == index_name:
                logger.info(f"Using existing index: {index_name}")
                return index.id
        
        logger.info(f"Creating index: {index_name}")
        index = client.index.create(
            name=index_name,
            models=[{"name": engine_name, "options": engine_options}]
        )
        return index.id
        
    except Exception as e:
        if "already_exists" in str(e):
            indexes = list(client.index.list())
            for index in indexes:
                if index.name == index_name:
                    return index.id
        raise e