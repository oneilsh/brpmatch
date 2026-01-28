"""Utilities for displaying colored warnings."""

import warnings
import sys


def warn(message: str, category=UserWarning, stacklevel: int = 2) -> None:
    """
    Issue a warning with color formatting for better visibility.

    Parameters
    ----------
    message : str
        Warning message to display
    category : Warning
        Warning category (default: UserWarning)
    stacklevel : int
        Stack level for warning origin (default: 2)
    """
    # ANSI color codes - yellow for warnings
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    # Issue the standard warning (for logging, filtering, etc.)
    warnings.warn(message, category, stacklevel=stacklevel + 1)

    # Also print a colored version to stderr if it's a TTY
    if sys.stderr.isatty():
        formatted_msg = f"{YELLOW}{BOLD}WARNING:{RESET} {YELLOW}{message}{RESET}"
        print(formatted_msg, file=sys.stderr)
