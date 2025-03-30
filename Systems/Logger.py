import os
from typing import Dict, List, Set, Optional, Any

class Logger:
    """
    Centralized logging system for the occlusion framework.
    Allows selective logging based on specified options.
    """
    
    # Singleton instance
    _instance = None
    
    # Available log categories
    SYSTEM = "system-logs"
    MODEL = "model-logs"
    RENDER = "render-logs"
    OCCLUSION = "occlusion-logs"
    DEBUG = "debug-logs"
    PERFORMANCE = "performance-logs"
    WARNING = "warning-logs"
    ERROR = "error-logs"
    
    # All available categories as a set for validation
    ALL_CATEGORIES = {SYSTEM, MODEL, RENDER, OCCLUSION, DEBUG, PERFORMANCE, WARNING, ERROR}
    
    def __new__(cls, *args, **kwargs):
        """Ensure singleton pattern"""
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, enabled_categories: Optional[List[str]] = None, log_to_file: bool = False, log_file_path: Optional[str] = None):
        """
        Initialize the logger with enabled categories.
        
        Args:
            enabled_categories: List of log categories to enable. If None, only errors are enabled.
            log_to_file: Whether to log to a file in addition to console
            log_file_path: Path to the log file. If None, logs are written to 'occlusion_framework.log'
        """
        # Skip initialization if already initialized
        if getattr(self, '_initialized', False):
            return
            
        # Set default enabled categories if none provided
        if enabled_categories is None:
            self.enabled_categories = {self.ERROR}
        else:
            # Validate and filter categories
            self.enabled_categories = {cat for cat in enabled_categories if cat in self.ALL_CATEGORIES}
            
        # Add ERROR category by default for safety
        self.enabled_categories.add(self.ERROR)
        
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
    
    def configure(self, enabled_categories: Optional[List[str]] = None, log_to_file: Optional[bool] = None, 
                 log_file_path: Optional[str] = None):
        """
        Reconfigure the logger.
        
        Args:
            enabled_categories: List of log categories to enable
            log_to_file: Whether to log to a file
            log_file_path: Path to the log file
        """
        if enabled_categories is not None:
            # Validate and filter categories
            self.enabled_categories = {cat for cat in enabled_categories if cat in self.ALL_CATEGORIES}
            # Always keep ERROR enabled
            self.enabled_categories.add(self.ERROR)
            
        if log_to_file is not None:
            self.log_to_file = log_to_file
            
        if log_file_path is not None:
            self.log_file_path = log_file_path
            
        # Create log file if needed
        if self.log_to_file and not os.path.exists(os.path.dirname(self.log_file_path)):
            os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
    
    def is_category_enabled(self, category: str) -> bool:
        """
        Check if a log category is enabled.
        
        Args:
            category: The category to check
            
        Returns:
            True if the category is enabled, False otherwise
        """
        return category in self.enabled_categories
    
    def log(self, message: str, category: str = SYSTEM):
        """
        Log a message if the category is enabled.
        
        Args:
            message: The message to log
            category: The category of the log message
        """
        if category not in self.ALL_CATEGORIES:
            # Invalid category, default to SYSTEM
            category = self.SYSTEM
            
        if category in self.enabled_categories:
            # Format the message with category prefix
            formatted_message = f"[{category}] {message}"
            
            # Print to console
            print(formatted_message)
            
            # Write to file if enabled
            if self.log_to_file:
                with open(self.log_file_path, 'a') as f:
                    f.write(f"{formatted_message}\n")
    
    def system(self, message: str):
        """Log a system message"""
        self.log(message, self.SYSTEM)
    
    def model(self, message: str):
        """Log a model-related message"""
        self.log(message, self.MODEL)
    
    def render(self, message: str):
        """Log a render-related message"""
        self.log(message, self.RENDER)
    
    def occlusion(self, message: str):
        """Log an occlusion-related message"""
        self.log(message, self.OCCLUSION)
    
    def debug(self, message: str):
        """Log a debug message"""
        self.log(message, self.DEBUG)
    
    def performance(self, message: str):
        """Log a performance-related message"""
        self.log(message, self.PERFORMANCE)
    
    def warning(self, message: str):
        """Log a warning message"""
        self.log(message, self.WARNING)
    
    def error(self, message: str):
        """Log an error message (always enabled)"""
        self.log(message, self.ERROR)

# Create a global logger instance with default settings
logger = Logger()