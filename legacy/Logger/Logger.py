import os
from typing import Dict, List, Set, Optional, Any

class Logger:
    """
    Centralized logging system for the occlusion framework.
    Allows selective logging based on specified options.
    """
    
    # Singleton instance
    _instance = None
    
    # Log key constants
    SYSTEM = "system"
    MODEL = "model"
    RENDER = "render"
    OCCLUSION = "occlusion"
    DEBUG = "debug"
    PERFORMANCE = "performance"
    WARNING = "warning"
    ERROR = "error"
    
    # All available log keys as a set for reference
    ALL_KEYS = {SYSTEM, MODEL, RENDER, OCCLUSION, DEBUG, PERFORMANCE, WARNING, ERROR}
    
    def __new__(cls, *args, **kwargs):
        """Ensure singleton pattern"""
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, enabled_log_keys: Optional[List[str]] = None, log_to_file: bool = False, log_file_path: Optional[str] = None):
        """
        Initialize the logger with enabled log keys.
        
        Args:
            enabled_log_keys: List of log keys to enable. If None, only errors are enabled.
            log_to_file: Whether to log to a file in addition to console
            log_file_path: Path to the log file. If None, logs are written to 'occlusion_framework.log'
        """
        # Skip initialization if already initialized
        if getattr(self, '_initialized', False):
            return
            
        # Set default enabled log keys if none provided
        if enabled_log_keys is None:
            self.enabled_log_keys = {self.ERROR}
        else:
            self.enabled_log_keys = set(enabled_log_keys)
            
        # Add ERROR key by default for safety
        self.enabled_log_keys.add(self.ERROR)
        
        # File logging settings
        self.log_to_file = log_to_file
        self.log_file_path = log_file_path or "occlusion_framework.log"
        
        # Create log file if needed
        if self.log_to_file:
            log_dir = os.path.dirname(self.log_file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # Clear the log file
            with open(self.log_file_path, 'w') as f:
                f.write(f"=== Occlusion Framework Log ===\n")
        
        self._initialized = True
    
    def configure(self, enabled_log_keys: Optional[List[str]] = None, log_to_file: Optional[bool] = None, 
                 log_file_path: Optional[str] = None):
        """
        Reconfigure the logger.
        
        Args:
            enabled_log_keys: List of log keys to enable
            log_to_file: Whether to log to a file
            log_file_path: Path to the log file
        """
        if enabled_log_keys is not None:
            self.enabled_log_keys = set(enabled_log_keys)
            # Always keep ERROR enabled
            self.enabled_log_keys.add(self.ERROR)
            
        if log_to_file is not None:
            self.log_to_file = log_to_file
            
        if log_file_path is not None:
            self.log_file_path = log_file_path
            
        # Create log file if needed
        if self.log_to_file and not os.path.exists(os.path.dirname(self.log_file_path)):
            os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
    
    def is_log_key_enabled(self, log_key: str) -> bool:
        """
        Check if a log key is enabled.
        
        Args:
            log_key: The log key to check
            
        Returns:
            True if the log key is enabled, False otherwise
        """
        return log_key in self.enabled_log_keys
    
    def log(self, log_key: str, message: str):
        """
        Log a message if the log key is enabled.
        
        Args:
            log_key: The log key for the message
            message: The message to log
        """
        if log_key in self.enabled_log_keys:
            # Format the message with log key prefix
            formatted_message = f"[{log_key}] {message}"
            
            # Print to console
            print(formatted_message)
            
            # Write to file if enabled
            if self.log_to_file:
                with open(self.log_file_path, 'a') as f:
                    f.write(f"{formatted_message}\n")

# Create a global logger instance with default settings
logger = Logger()