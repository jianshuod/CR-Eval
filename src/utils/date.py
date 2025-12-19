from dateutil import parser
from src.utils.logging import logging

logger = logging.getLogger(__name__)


def parse_and_format_date(date_str):
    try:
        parsed_date = parser.parse(date_str)
        formatted_date = parsed_date.strftime("%Y-%m-%d")
        return formatted_date
    except ValueError:
        return None
