# Occlusion Framework Logging System

## Overview

The Occlusion Framework Logging System provides a centralized, flexible, and configurable logging mechanism for the entire application. It replaces direct `print()` calls with a structured logging approach that allows selective enabling/disabling of different log categories.

The Logger is implemented as a standalone module in the `Logger` directory, making it easily accessible from anywhere in the project.

## Features

- **Key-based Logging**: Logs are organized by keys for better filtering and organization
- **Selective Enabling**: Enable only the log keys you need
- **Console and File Output**: Option to log to both console and file
- **Singleton Pattern**: Consistent logging behavior across the application
- **Command-line Configuration**: Easy configuration through command-line arguments
- **Runtime Reconfiguration**: Change logging behavior during runtime

## Log Keys

The system provides the following predefined log keys as constants:

| Key | Constant | Description |
|-----|----------|-------------|
| system | `Logger.SYSTEM` | General system operations and workflow |
| model | `Logger.MODEL` | Model loading and processing |
| render | `Logger.RENDER` | Rendering operations |
| occlusion | `Logger.OCCLUSION` | Occlusion mask generation |
| debug | `Logger.DEBUG` | Detailed debugging information |
| performance | `Logger.PERFORMANCE` | Performance metrics and timing |
| warning | `Logger.WARNING` | Warning messages |
| error | `Logger.ERROR` | Error messages (always enabled by default) |

## Usage

### Basic Usage

```python
from Logger import logger, Logger

# Log messages with different keys
logger.log(Logger.SYSTEM, "System is initializing")
logger.log(Logger.MODEL, "Loading model: model.fbx")
logger.log(Logger.RENDER, "Rendering frame 42")
logger.log(Logger.OCCLUSION, "Generating occlusion mask")
logger.log(Logger.DEBUG, "Variable x = 42")
logger.log(Logger.PERFORMANCE, "Operation completed in 0.5s")
logger.log(Logger.WARNING, "Resource usage is high")
logger.log(Logger.ERROR, "Failed to load file: file.obj")
```

### Configuration

#### Through Command Line

When using the BaseSystem's command-line interface:

```bash
python main.py --log-keys system model error --log-to-file --log-file logs/app.log
```

#### Programmatically

```python
from Logger import logger, Logger

# Configure at initialization
logger = Logger(
    enabled_log_keys=[Logger.SYSTEM, Logger.ERROR],
    log_to_file=True,
    log_file_path="logs/app.log"
)

# Or reconfigure later
logger.configure(
    enabled_log_keys=[Logger.SYSTEM, Logger.MODEL, Logger.RENDER, Logger.ERROR],
    log_to_file=True,
    log_file_path="logs/detailed.log"
)
```

#### In Example Application

When using the example.py:

```
Enter log options (comma-separated, leave empty for default) > system,model,render
```

### Checking if a Key is Enabled

```python
if logger.is_log_key_enabled(Logger.DEBUG):
    # Perform expensive debug operations
    detailed_debug_info = generate_detailed_debug_info()
    logger.log(Logger.DEBUG, detailed_debug_info)
```

## Implementation Details

### Logger Class

The `Logger` class is implemented as a singleton to ensure consistent logging behavior across the application. It provides:

- A generic `log(log_key, message)` method for all logging
- Predefined constants for common log keys
- Configuration methods
- Key validation
- File logging capabilities

### Integration with BaseSystem

The BaseSystem class is integrated with the logging system and provides command-line options for configuring logging behavior.

### Default Behavior

By default, only error logs are enabled to minimize output. The ERROR key is always enabled for safety, even if not explicitly specified.

## Extending the System

### Adding New Log Keys

To add a new log key:

1. Add a new constant to the Logger class
2. Add the constant to the ALL_KEYS set

Example:

```python
# In Logger.py
class Logger:
    # Existing keys...
    NETWORK = "network"
    
    # Update ALL_KEYS
    ALL_KEYS = {SYSTEM, MODEL, RENDER, OCCLUSION, DEBUG, PERFORMANCE, WARNING, ERROR, NETWORK}
```

Then you can use it like any other key:

```python
logger.log(Logger.NETWORK, "Network connection established")
```

## Best Practices

1. **Use Constants**: Always use the predefined constants (e.g., `Logger.SYSTEM`) instead of string literals
2. **Be Consistent**: Use the same key for similar types of messages
3. **Check Key Status**: For expensive logging operations, check if the key is enabled first
4. **Include Context**: Provide enough context in log messages to understand what's happening
5. **Use Structured Data**: For complex data, consider formatting as JSON or another structured format