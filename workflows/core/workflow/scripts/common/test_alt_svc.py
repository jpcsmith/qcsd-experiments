"""Test cases for the alt_svc module."""
import pytest
from common.alt_svc import parse


def test_parse():
    """Test the alt service parsing."""
    assert parse("clear") == []

    assert parse(
        'h3-29=":443"; ma=2592000,h3-T051=":443"; ma=2592000; persist=1,'
        'h3-Q050=":443"; ma=2592000,h3-Q046=":443"; ma=2592000,'
        'h3-Q043=":443"; ma=2592000,quic=":443"; ma=2592000; v="46,43"'
    ) == [
        ("h3-29", ":443"), ("h3-T051", ":443"), ("h3-Q050", ":443"),
        ("h3-Q046", ":443"), ("h3-Q043", ":443"), ("GQUIC046", ":443"),
        ("GQUIC043", ":443"),
    ]

    with pytest.raises(ValueError):
        parse('h3-27=:443;ma=2592000,h3-25=:443;ma=2592000,'
              'h3-Q050=:443;ma=2592000,h3-Q049=:443;ma=2592000,'
              'h3-Q048=:443;ma=2592000,h3-Q046=:443;ma=2592000,'
              'h3-Q043=:443;ma=2592000,quic=:443;ma=2592000;v=46,43,',
              errors="raise")
