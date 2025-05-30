import sys
import logging


def log_to_file(path, log=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if log:
                # Set up logging to both file and terminal
                logger = logging.getLogger(func.__name__)
                logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels

                # Create a file handler for logging to the specified file
                file_handler = logging.FileHandler(path)
                file_handler.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels
                file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(file_formatter)

                # Create a stream handler for logging to the terminal
                stream_handler = logging.StreamHandler(sys.stdout)
                stream_handler.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels
                stream_formatter = logging.Formatter('%(levelname)s - %(message)s')
                stream_handler.setFormatter(stream_formatter)

                # Add both handlers to the logger
                logger.addHandler(file_handler)
                logger.addHandler(stream_handler)

                # Redirect stdout to a custom stream that writes to both file and terminal
                class DualStream:
                    def __init__(self, file_handler, stream_handler):
                        self.file_handler = file_handler
                        self.stream_handler = stream_handler

                    def write(self, message):
                        if message.strip():  # Avoid logging empty messages
                            # Determine the log level based on the message content
                            if "ERROR" in message:
                                level = logging.ERROR
                            elif "WARNING" in message:
                                level = logging.WARNING
                            else:
                                level = logging.INFO

                            # Write to file
                            self.file_handler.emit(logging.LogRecord(
                                name=func.__name__,
                                level=level,
                                pathname=__file__,
                                lineno=0,
                                msg=message.strip(),
                                args=None,
                                exc_info=None
                            ))

                        # Always write to terminal
                        self.stream_handler.stream.write(message)

                    def flush(self):
                        self.stream_handler.stream.flush()

                # Replace sys.stdout with the custom DualStream
                original_stdout = sys.stdout
                sys.stdout = DualStream(file_handler, stream_handler)

                try:
                    # Call the original function
                    result = func(*args, **kwargs)
                    logger.info(f"Function '{func.__name__}' executed successfully.")
                    return result
                except Exception as e:
                    logger.error(f"Function '{func.__name__}' raised an exception: {e}")
                    raise
                finally:
                    # Restore the original stdout
                    sys.stdout = original_stdout
                    # Remove handlers to avoid duplicate logs in future calls
                    logger.removeHandler(file_handler)
                    logger.removeHandler(stream_handler)
            else:
                # If log is False, just call the function normally
                return func(*args, **kwargs)

        return wrapper
    return decorator