import logging
import sys
from stormpy import Rational

logger = logging.getLogger(__name__)


def const(is_exact: bool, val: float):
    if is_exact:
        return Rational(val)
    else:
        return val


class MultiLineFormatter(logging.Formatter):
    """Multi-line formatter."""

    def get_header_length(self, record):
        """Get the header length of a given record."""
        rec = logging.LogRecord(
            name=record.name,
            level=record.levelno,
            pathname=record.pathname,
            lineno=record.lineno,
            msg="",
            args=(),
            exc_info=None,
        )
        return len(super().format(rec))

    def format(self, record):
        """Format a record with added indentation."""
        indent = " " * self.get_header_length(record)
        head, *trailing = super().format(record).splitlines(True)
        return head + "".join(indent + line for line in trailing)


def setup_logging():
    global logger

    logger.setLevel(logging.DEBUG)
    print(logger)
    handler = logging.StreamHandler(sys.stdout)
    formatter = MultiLineFormatter(
        "%(levelname)s:%(asctime)s - %(filename)s:%(lineno)d - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(handler)
    print(logger.handlers)
