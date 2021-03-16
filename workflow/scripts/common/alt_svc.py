"""Logic for working with HTTP AltSvc header fields."""
import re
import logging
from typing import List, NamedTuple


Record = NamedTuple('Record', [('protocol', str), ('authority', str)])
_REGEX = re.compile(
    r'(?:(?P<gproto>quic)|(?P<protocol>[a-zA-Z0-9-]+))="(?P<auth>.+?)"'
    r'(?P<ma>;ma=\d+|)?'
    r'(?(ma);persist=1|)?'
    r'(?(gproto);v="(?P<gvers>[\d,]+)"|)'
    r';?'
)


def parse(alt_svc: str, errors="raise") -> List[Record]:
    """Parse and return a dict of protocol -> authorities from the
    alt_svc string.  Errors can either be "raise" or "log" to raise
    or log parsing failure respectively.
    """
    result: List[Record] = []
    alt_svc = alt_svc.replace(" ", "")
    if alt_svc == "clear":
        return []

    match = _REGEX.match(alt_svc)
    while match:
        assert match.span()[0] == 0
        assert match.group("auth") is not None

        if match.group("gvers") is not None:
            for gver in map(int, match.group("gvers").split(",")):
                result.append(Record(f"GQUIC{gver:03d}", match.group("auth")))
        else:
            result.append(Record(match.group("protocol"), match.group("auth")))

        # Slice the alt_svc string to remove the match and (possible) comma
        alt_svc = alt_svc[(match.span()[1]+1):]
        match = _REGEX.match(alt_svc)

    # Check if we successfully parsed everything
    if alt_svc != "":
        assert errors in ("raise", "log")
        if errors == "raise":
            raise ValueError(f"Unable to finish parsing {alt_svc}, {result}")
        logging.getLogger(__name__).warning(
            "Unable to finish parsing %s, progress: %s", alt_svc, result)
    return result
