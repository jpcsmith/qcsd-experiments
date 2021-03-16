"""Tests for the filter module."""
import pytest
import numpy as np

from .filter_versions import (
    parse_alt_svc, remove_similar_roots, reduce_representation
)


def test_parse_alt_svc():
    """It should correctly parse alt svc records."""
    record = ('h3-29=":443"; ma=2592000,h3-T051=":443"; ma=2592000,'
              'h3-Q050=":443"; ma=2592000')
    assert parse_alt_svc(record, "example.com") == [
        ("h3-29", "example.com:443"), ("h3-T051", "example.com:443"),
        ("h3-Q050", "example.com:443")
    ]


def test_parse_alt_svc_host_auth():
    """It should correctly parse alt svc records."""
    record = 'h3-29=":443",h3-T051="alt.example.com:443"'
    assert parse_alt_svc(record, "example.com") == [
        ("h3-29", "example.com:443"), ("h3-T051", "alt.example.com:443"),
    ]


def test_parse_alt_svc_gquic():
    """It should ignore the non-standard 'v="..."' label that google
    uses.
    """
    record = ('h3-29=":443"; ma=2592000,h3-T051=":443"; ma=2592000,'
              'h3-Q050=":443"; ma=2592000,h3-Q046=":443"; ma=2592000,'
              'h3-Q043=":443"; ma=2592000,quic=":443"; ma=2592000; v="46,43"')
    assert parse_alt_svc(record, "example.com") == [
        ("h3-29", "example.com:443"), ("h3-T051", "example.com:443"),
        ("h3-Q050", "example.com:443"), ("h3-Q046", "example.com:443"),
        ("h3-Q043", "example.com:443"), ("quic", "example.com:443")
    ]


def test_remove_similar_roots():
    """It should filter out URLs which differ in only private parts,
    prefering to keep entries with a lower rank.
    """
    np.testing.assert_array_equal(
        remove_similar_roots(
            ["b.com", "b.co.uk",   "a.com", "a.de",     "a.b.com"],
            ranks=[4, 3, 2, 1, 5]),
        [1, 3, 4])


def test_remove_similar_roots_with_port():
    """It should filter ignoring ports."""
    np.testing.assert_array_equal(
        remove_similar_roots(
            ["b.com:443", "b.co.uk:553",    "a.com:663", "a.de:773",
             "a.b.com:883"],
            ranks=[4, 3, 2, 1, 5]),
        [1, 3, 4])


@pytest.mark.parametrize(
    "threshold,expected", [(1, [0, 1, 4]), (2, [0, 1, 2, 4, 5]),
                           (3, [0, 1, 2, 3, 4, 5])]
)
def test_reduce_representation(threshold, expected):
    """It should remove domains with a similar private parts of their
    names.
    """
    np.testing.assert_array_equal(
        reduce_representation(
            ["first.apple.com",
             "first.ball.com", "second.ball.com", "third.ball.com",
             "first.cat.com", "second.cat.com"],
            sld_domains=["ball.com", "cat.com"], threshold=threshold,
            ranks=list(range(1, 7))),
        expected)
