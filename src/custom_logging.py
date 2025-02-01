import os
import logging # debug, info, warning, error, critical
from typing import Optional, Union

class CustomLogger:
    def __init__(self, name:str=__name__, level:Union[int, str]='warning', print_to_console:bool=True):
        self._logger = logging.getLogger(name)
        self.set_level(level)

        self._console_handler = logging.StreamHandler()
        self._console_handler.setFormatter(logging.Formatter(self.default_format))
        self._console_handler.setLevel(self._logger.level)
        if print_to_console:
            self._logger.addHandler(self._console_handler)

        self._print_to_console = print_to_console

    @property
    def print_to_console(self):
        return self._print_to_console
    
    @print_to_console.setter
    def print_to_console(self, value: bool):
        if value and (self._console_handler not in self._logger.handlers):
            self._logger.addHandler(self._console_handler)
        elif not value and (self._console_handler in self._logger.handlers):
            self._logger.removeHandler(self._console_handler)

    @property
    def default_format(self):
        return "[%(name)s | %(levelname)s] %(asctime)s - %(message)s"
    
    def log(self, level:Union[int, str], message:str):
        """Log a message at the specified level.

        Args:
            level: The logging level (int or str).
            message: The message to log.
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.WARNING)
        self._logger.log(level, message)
    
    def debug(self, message:str):
        """Log a debug message."""
        self.log(logging.DEBUG, message)

    def info(self, message:str):
        """Log an info message."""
        self.log(logging.INFO, message)

    def warning(self, message:str):
        """Log a warning message."""
        self.log(logging.WARNING, message)

    def error(self, message:str):
        """Log an error message."""
        self.log(logging.ERROR, message)

    def critical(self, message:str):
        """Log a critical message."""
        self.log(logging.CRITICAL, message)

    def exception(self, message:str):
        """Log an exception message."""
        self._logger.exception(message)


    def set_level(self, level: Union[int, str]):
        """Set the logging level for the logger.

        Args:
            level: The logging level to set. Can be an integer (through logging) or a string.
        """
        if isinstance(level, str):
            if level.upper() not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
                raise ValueError(f"Invalid log level: {level}")
            level = getattr(logging, level.upper(), logging.WARNING)
        self._logger.setLevel(level)


    def add_handler(self, handler: logging.Handler):
        """Add a custom pre-set handler to the logger.

        Args:
            handler: A logging.Handler instance to add to the logger.
        """
        self._logger.addHandler(handler)

    def add_file_handler(self, file_path: str, mode:str='a', level:Optional[Union[int, str]]=None, logging_format:Optional[str]=None):
        """Add a file handler to the logger.

        Args:
            mode: Recommended to use 'w' for new logs and 'a' for appending logs. Defaults to 'a'.
            level: The logging level for the file handler. If None, uses the logger's level.
            logging_format: The format for the file handler. If None, uses the default format.
        """
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True)
        new_handler = logging.FileHandler(file_path, mode=mode)
        if logging_format:
            new_handler.setFormatter(logging.Formatter(logging_format))
        else:
            new_handler.setFormatter(logging.Formatter(self.default_format))
        if level:
            new_handler.setLevel(level)
        else:
            new_handler.setLevel(self._logger.level)
        self._logger.addHandler(new_handler)

    def remove_handlers(self, types:Optional[Union[str, list[str]]]=None):
        """Remove one or more specific types of handlers from the logger.
        
        Args:
            types: A single handler type as a string or a list of handler types to remove. If None, remove all handlers.
        """
        if types is None:
            for handler in self._logger.handlers[:]:
                self._logger.removeHandler(handler)
            return
        
        if isinstance(types, str):
            types = [types]
        types = [t.lower() for t in types]

        for handler in self._logger.handlers[:]:
            if handler.__class__.__name__.lower() in types:
                self._logger.removeHandler(handler) 
















