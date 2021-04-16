"""Parse QUIC packets from PCAPS"""
import json
import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import List, Iterator, NamedTuple, Set

_LOGGER = logging.getLogger(__name__)


@dataclass
class StreamFrame:
    """A QUIC stream frame."""
    details: dict

    @property
    def stream_id(self) -> int:
        """The stream ID of the stream."""
        return int(self.details["quic.stream.stream_id"])

    def data_length(self) -> int:
        """The length of the stream frame's data."""
        # The data is encoded as colon separated hex values, here we calculate
        # the byte length
        char_length = len(self.details["quic.stream_data"])
        if char_length == 0:
            return 0

        assert (char_length + 1) % 3 == 0
        return (char_length + 1) // 3


def _unpack_udp_payload(data: dict) -> Iterator[dict]:
    """Some UDP packets have multiple (or what what wireshark believes
    to be multiple) QUIC packets. Return each QUIC packet singly wrapped
    """
    layers = data["_source"]["layers"]
    quic_datagrams = layers["quic"]
    if not isinstance(quic_datagrams, list):
        quic_datagrams = [quic_datagrams, ]

    results: List[dict] = []
    for datagram in quic_datagrams:
        if "quic.coalesced_padding_data" in datagram.get("_ws.expert", {}):
            # Padding of the previous dgram, cant easily get the length
            assert len(results) > 0
            continue

        if (
            "quic.remaining_payload" in datagram
            and not datagram["quic.remaining_payload"].strip(":0")
        ):
            # This is a "packet" of all zeroes that was padding the previous
            assert len(results) > 0
            results[-1]["other_padding"] += int(datagram["quic.packet_length"])
            continue

        datagram.update({
            "frame.number": layers["frame"]["frame.number"],
            "frame.time_epoch": layers["frame"]["frame.time_epoch"],
            "udp.srcport": layers["udp"]["udp.srcport"],
            "udp.dstport": layers["udp"]["udp.dstport"],
            "other_padding": 0,
        })
        results.append(datagram)

    yield from results


class UndecryptedPacketError(ValueError):
    """Raised when a QUIC packet was not decrypted."""


class UndecryptedTraceError(ValueError):
    """Raised when a QUIC trace appears to not be decrypted."""


@dataclass(init=False)
class QuicPacket:
    """A QUIC packet based on tshark's JSON output.

    The underlying json representation can be created with:

        tshark -Tjson --no-duplicate-keys -J"frame udp quic"

    """
    details: dict

    def __init__(self, data: dict):
        if "quic.decryption_failed" in data.get("_ws.expert", {}):
            raise UndecryptedPacketError(
                data["_ws.expert"]["_ws.expert.message"])

        if "quic.remaining_payload" in data:
            raise UndecryptedPacketError("Packet has remaining payload")

        if "quic.packet_length" not in data:
            raise ValueError(f"Packet has no packet_length: {data}")

        self.details = data

        # Make quic.frame always be a list
        frames = self.details.setdefault("quic.frame", [])
        if not isinstance(frames, list):
            self.details["quic.frame"] = [frames]

    @property
    def timestamp(self) -> float:
        """Time since epoch (frame.time_epoch)"""
        return float(self.details["frame.time_epoch"])

    @property
    def packet_number(self) -> int:
        """QUIC packet number (quic.packet_number)"""
        if "quic.packet_number" in self.details:
            return int(self.details["quic.packet_number"])
        if "quic.short" in self.details:
            return int(self.details["quic.short"]["quic.packet_number"])

        raise NotImplementedError("Unable to get packet number")

    @property
    def packet_length(self) -> int:
        """The total length of the QUIC packet."""
        return int(self.details["quic.packet_length"])

    def is_outgoing(self) -> bool:
        """Returns true iff the packet is an outgoing packet, i.e. one
        destined for port 443.
        """
        return self.details["udp.srcport"] != "443"

    def frames(self, filter_=None) -> List[dict]:
        """Return the frames in the packet."""
        if filter_ is None:
            return self.details["quic.frame"]
        return [frame for frame in self.frames() if filter_(frame)]

    def stream_frames(self) -> List[StreamFrame]:
        """A helper that returns stream frame objects instead of dicts."""
        return [StreamFrame(frame) for frame in self.frames(
            lambda x: "quic.stream.stream_id" in x)]

    def padding(self) -> int:
        """The total amount QUIC padding added to the packet."""
        return self.details["other_padding"] + sum(
            int(frame["quic.padding_length"]) for frame in
            self.frames(lambda f: f["quic.frame_type"] == "0")
        )


def to_quic_packets(pcap_file: str, display_filter: str = "") \
        -> Iterator[QuicPacket]:
    """Parse QUIC packets from the specified pcap file"""
    command = ["tshark", "-r", pcap_file, "-T", "json", "-J", "frame udp quic",
               "--no-duplicate-keys"]
    if display_filter:
        command += ["-Y", display_filter]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    assert result.stdout is not None
    _LOGGER.debug("Subprocess call: %s", result.args)

    # Count the number of sequential undecrypted packets
    sequential_undecrypted = 0

    for pkt in json.loads(result.stdout):
        for entry in _unpack_udp_payload(pkt):
            try:
                yield QuicPacket(entry)
                sequential_undecrypted = 0
            except UndecryptedPacketError:
                sequential_undecrypted += 1

            if sequential_undecrypted == 10:
                raise UndecryptedTraceError(
                    f"Too many encrypted packets: {sequential_undecrypted}"
                )


class QuicPacketSummary(NamedTuple):
    """A tuple summarising the data within a QUIC packet.

    Attributes:
        timestamp: The time at which the packet was captured
        length: The total length of the QUIC packet, excluding UDP/IP headers
        length_chaff: The total length of chaff streams and padding
        length_app_streams: The total length of non-chaff streams
        is_outgoing: True iff the source port of the packet was not port 443
        number: The QUIC packet number
    """
    time: float
    length: int
    length_chaff: int
    length_app_streams: int
    is_outgoing: bool
    number: int


def _load_chaff_stream_ids(chaff_streams_file: str) -> Set[int]:
    lines = Path(chaff_streams_file).read_text().split("\n")
    try:
        # Assume one stream_id per line and see if we can parse it
        return set(int(stream_id) for stream_id in lines)
    except ValueError:
        pass

    # Then this may be a stdout file, in which case attempt to parse
    # The format of lines with chaff stream ids in the stdout is:
    #   Successfully created new dummy stream id <id> for resource ...
    token = "Successfully created new dummy stream id "
    # Filter to lines with the token and drop everything before the ID
    lines = [line[len(token):] for line in lines if line.startswith(token)]
    # Extract the ID and convert to an integer
    return set(int(line.split(maxsplit=1)[0]) for line in lines)


def to_quic_lengths(
    pcap_file: str, chaff_streams_file: str,
) -> Iterator[QuicPacketSummary]:
    """Parse only chaff traffic from the pcap_file and file specifying
    chaff streams.  Only the details of packets with chaff traffic are
    returned.
    """
    chaff_streams = _load_chaff_stream_ids(chaff_streams_file)

    for pkt in to_quic_packets(pcap_file):
        yield QuicPacketSummary(
            time=pkt.timestamp,
            length=pkt.packet_length,
            length_chaff=pkt.padding() + sum(
                frame.data_length() for frame in pkt.stream_frames()
                if frame.stream_id in chaff_streams
            ),
            length_app_streams=sum(
                frame.data_length() for frame in pkt.stream_frames()
                if frame.stream_id not in chaff_streams
            ),
            is_outgoing=pkt.is_outgoing(),
            number=pkt.packet_number,
        )
