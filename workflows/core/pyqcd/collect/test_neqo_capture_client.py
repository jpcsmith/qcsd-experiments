from ipaddress import IPv4Address, IPv6Address

from .neqo_capture_client import extract_endpoints


def test_extract_endpoints_ipv4():
    """It should extract IPv4 endpoints."""
    text = "\n".join([
        "Cannot create stream Err(Unavailable)",
        ("H3 Client connecting: "
         "V4(192.33.90.80:47671) -> V4(172.217.168.68:443)"),
        "n_c: 6  w_c: 1.4484675414313228",
        "n_s: 810        w_s: 1.9405179046383045",
        "Cannot create stream Err(Unavailable)",
        "Cannot create dummy stream Err(Unavailable)",
    ])

    (local, remote) = extract_endpoints(text)
    assert local == (IPv4Address("192.33.90.80"), 47671)
    assert remote == (IPv4Address("172.217.168.68"), 443)


def test_extract_endpoints_default_version():
    """It should extract IPv4 endpoints when no version is specified."""
    text = "\n".join([
        "H3 Client connecting: 192.33.90.80:47671 -> 172.217.168.68:443",
        "n_c: 6  w_c: 1.4484675414313228",
        "n_s: 810        w_s: 1.9405179046383045",
        "Cannot create stream Err(Unavailable)",
        "Cannot create dummy stream Err(Unavailable)",
    ])

    (local, remote) = extract_endpoints(text)
    assert local == (IPv4Address("192.33.90.80"), 47671)
    assert remote == (IPv4Address("172.217.168.68"), 443)


def test_extract_endpoints_ipv6():
    """It should extract IPv6 endpoints."""
    text = "\n".join([
        ("H3 Client connecting: "
         "V6([2001:1711:fa55:b9d0:4926:da7d:8e50:57bb]:51957) "
         "-> V6([2a00:1450:400a:800::2004]:443)"),
        "n_c: 692	w_c: 1.4603483901900693",
        "n_s: 790	w_s: 2.4819836498281527",
        "Cannot create stream Err(Unavailable)",
        "Cannot create dummy stream Err(Unavailable)",
    ])

    (local, remote) = extract_endpoints(text)
    assert local == (
        IPv6Address("2001:1711:fa55:b9d0:4926:da7d:8e50:57bb"), 51957)
    assert remote == (IPv6Address("2a00:1450:400a:800::2004"), 443)
