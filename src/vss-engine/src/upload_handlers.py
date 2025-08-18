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
"""Generic upload handler registry for VSS asset uploads."""

import os
import importlib
from typing import List, Dict, Callable, Any
from via_logger import logger


class UploadHandlerRegistry:
    """Registry for asset upload handlers."""
    
    def __init__(self):
        self.handlers: List[Dict[str, Any]] = []
    
    def register_handler(self, name: str, handler: Callable, filter_func: Callable = None):
        """Register an upload handler.
        
        Args:
            name: Human-readable name for the handler
            handler: Function to call for uploads (takes asset as parameter)
            filter_func: Function to determine if handler should process asset (optional)
        """
        self.handlers.append({
            'name': name,
            'handler': handler,
            'filter': filter_func or (lambda asset: True)
        })
        logger.info(f"Registered upload handler: {name}")
    
    def get_upload_callback(self):
        """Get a unified callback function for all registered handlers."""
        if not self.handlers:
            return None
        
        def upload_callback(asset):
            for handler_config in self.handlers:
                if handler_config['filter'](asset):
                    try:
                        handler_config['handler'](asset)
                    except Exception as e:
                        logger.error(f"[{handler_config['name']}] Upload handler failed: {e}")
        
        return upload_callback
    
    def load_handlers_from_config(self):
        """Load handlers based on environment configuration."""
        # Default handler modules to try
        default_modules = [
            "models.twelve_labs.twelve_labs_upload_handler",
            # Add other default handlers here
        ]
        
        # Additional modules from environment
        handler_modules = os.getenv("VSS_UPLOAD_HANDLER_MODULES", "").split(",")
        all_modules = default_modules + [m.strip() for m in handler_modules if m.strip()]
        
        for module_name in all_modules:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, 'register_upload_handlers'):
                    module.register_upload_handlers(self)
                    logger.info(f"Loaded upload handlers from: {module_name}")
            except Exception as e:
                logger.debug(f"Could not load upload handlers from {module_name}: {e}")


# Global registry instance
upload_registry = UploadHandlerRegistry()