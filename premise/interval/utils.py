import datetime
import logging
import sys
from stormpy import Rational

logger = logging.getLogger(__name__)


def const(is_exact: bool, val: float):
    if is_exact:
        return Rational(val)
    else:
        return val


class TimeFilter(logging.Filter):

    def filter(self, record):
        if record.levelno == logging.DEBUG + 1:
            record.relative = ""
            return True

        try:
            last = self.last
        except AttributeError:
            last = record.relativeCreated

        delta = datetime.datetime.fromtimestamp(
            record.relativeCreated / 1000.0
        ) - datetime.datetime.fromtimestamp(last / 1000.0)

        record.relative = "{0:.2f}".format(
            delta.seconds + delta.microseconds / 1000000.0
        )

        self.last = record.relativeCreated
        return True


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
        rec.relative = record.relative
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
        "%(levelname)s:%(asctime)s - (%(relative)ss) - %(filename)s:%(lineno)d - %(message)s"
    )
    time_filter = TimeFilter()
    handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.addFilter(time_filter)
    print(logger.handlers)
