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
from via_logger import logger

try:
    from .twelve_labs_model import TwelveLabsModel
except ImportError:
    TwelveLabsModel = None

def register_upload_handlers(registry):
    if os.getenv("VSS_ENABLE_TWELVE_LABS_AUTO_UPLOAD", "false").lower() != "true":
        return
        
    if TwelveLabsModel is None:
        logger.warning("TwelveLabsModel not available")
        return
        
    try:
        model = TwelveLabsModel()
        
        async def upload_handler(asset):
            try:
                # Use the new async method that doesn't block
                result = await model.ensure_video_uploaded_async(asset.asset_id)
                if result.get("error"):
                    logger.error(f"Upload failed: {result.get('text')}")
                else:
                    logger.info("Upload successful")
            except Exception as e:
                logger.error(f"Upload error: {e}")
                # Don't re-raise to avoid crashing the background task
        
        registry.register_handler(
            name="Twelve Labs",
            handler=upload_handler,
            filter_func=lambda asset: asset.media_type == "video" and not asset.is_live
        )
        
    except Exception as e:
        logger.warning(f"Failed to initialize upload handler: {e}")