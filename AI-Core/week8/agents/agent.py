import logging

class Agent:
    """
    An abstract superclass for Agents
    Used to log messages in a way that can identify each Agent

    Each subclass sets `name` and `color`; log() prefixes every message with the name
    and wraps it in the colour, so in a terminal (or the Day 5 UI, via log_utils.reformat)
    you can see at a glance which agent is talking.
    """

    # Foreground colors (ANSI escape codes)
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Background color
    BG_BLACK = '\033[40m'

    # Reset code to return to default color
    RESET = '\033[0m'

    name: str = ""                          # shown in [brackets] before each log line
    color: str = '\033[37m'                 # foreground colour of this agent's log lines

    def log(self, message):
        """
        Log this as an info message, identifying the agent
        """
        color_code = self.BG_BLACK + self.color
        message = f"[{self.name}] {message}"
        logging.info(color_code + message + self.RESET)

