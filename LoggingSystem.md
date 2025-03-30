# Occlusion Framework Logging System

## Overview

The Occlusion Framework Logging System provides a centralized, flexible, and configurable logging mechanism for the entire application. It replaces direct `print()` calls with a structured logging approach that allows selective enabling/disabling of different log categories.

## Features

- **Categorized Logging**: Logs are organized into distinct categories for better filtering and organization
- **Selective Enabling**: Enable only the log categories you need
- **Console and File Output**: Option to log to both console and file
- **Singleton Pattern**: Consistent logging behavior across the application
- **Command-line Configuration**: Easy configuration through command-line arguments
- **Runtime Reconfiguration**: Change logging behavior during runtime

## Log Categories

The system supports the following log categories:

| Category | Constant | Description |
|----------|----------|-------------|
| System | `SYSTEM` | General system operations and workflow |
| Model | `MODEL` | Model loading and processing |
| Render | `RENDER` | Rendering operations |
| Occlusion | `OCCLUSION` | Occlusion mask generation |
| Debug | `DEBUG` | Detailed debugging information |
| Performance | `PERFORMANCE` | Performance metrics and timing |
| Warning | `WARNING` | Warning messages |
| Error | `ERROR` | Error messages (always enabled by default) |

## Usage

### Basic Usage

```python
from Systems.Logger import logger

# Log messages to different categories
logger.system("System is initializing")
logger.model("Loading model: model.fbx")
logger.render("Rendering frame 42")
logger.occlusion("Generating occlusion mask")
logger.debug("Variable x = 42")
logger.performance("Operation completed in 0.5s")
logger.warning("Resource usage is high")
logger.error("Failed to load file: file.obj")
```

### Configuration

#### Through Command Line

When using the BaseSystem's command-line interface:

```bash
python main.py --log-options system-logs model-logs error-logs --log-to-file --log-file logs/app.log
```

#### Programmatically

```python
from Systems.Logger import logger

# Configure at initialization
logger = Logger(
    enabled_categories=["system-logs", "error-logs"],
    log_to_file=True,
    log_file_path="logs/app.log"
)

# Or reconfigure later
logger.configure(
    enabled_categories=["system-logs", "model-logs", "render-logs", "error-logs"],
    log_to_file=True,
    log_file_path="logs/detailed.log"
)
```

#### In Example Application

When using the example.py:

```
Enter log options (comma-separated, leave empty for default) > system-logs,model-logs,render-logs
```

### Checking if a Category is Enabled

```python
if logger.is_category_enabled(logger.DEBUG):
    # Perform expensive debug operations
    detailed_debug_info = generate_detailed_debug_info()
    logger.debug(detailed_debug_info)
```

## Implementation Details

### Logger Class

The `Logger` class is implemented as a singleton to ensure consistent logging behavior across the application. It provides:

- Category-specific logging methods
- Configuration methods
- Category validation
- File logging capabilities

### Integration with BaseSystem

The BaseSystem class is integrated with the logging system and provides command-line options for configuring logging behavior.

### Default Behavior

By default, only error logs are enabled to minimize output. The ERROR category is always enabled for safety, even if not explicitly specified.

## Extending the System

### Adding New Categories

To add a new log category:

1. Add a new constant to the Logger class
2. Add the constant to the ALL_CATEGORIES set
3. Add a convenience method for the new category

Example:

```python
# In Logger.py
class Logger:
    # Existing categories...
    NETWORK = "network-logs"
    
    # Update ALL_CATEGORIES
    ALL_CATEGORIES = {SYSTEM, MODEL, RENDER, OCCLUSION, DEBUG, PERFORMANCE, WARNING, ERROR, NETWORK}
    
    # Add convenience method
    def network(self, message: str):
        """Log a network-related message"""
        self.log(message, self.NETWORK)
```

## Best Practices

1. **Use Appropriate Categories**: Choose the most specific category for each log message
2. **Be Consistent**: Use the same category for similar types of messages
3. **Check Category Status**: For expensive logging operations, check if the category is enabled first
4. **Include Context**: Provide enough context in log messages to understand what's happening
5. **Use Structured Data**: For complex data, consider formatting as JSON or another structured format