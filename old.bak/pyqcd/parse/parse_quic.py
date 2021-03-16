"""Parse QUIC packets from PCAPS"""
import copy
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterator, NamedTuple

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


@dataclass(init=False)
class QuicPacket:
    """A QUIC packet based on tshark's JSON output.

    The underlying json representation can be created with:

        tshark -Tjson --no-duplicate-keys -J"frame udp quic"

    """
    details: dict

    def __init__(self, data: dict):
        self.details = data["_source"]["layers"] if "_source" in data else data

        # Ensure that we do have a decrypted QUIC packet
        assert "quic.packet_length" in self.details["quic"]
        # Make quic.frame always be a list
        frames = self.details["quic"].setdefault("quic.frame", [])
        if not isinstance(frames, list):
            self.details["quic"]["quic.frame"] = [frames]

    @classmethod
    def from_json(cls, data: dict) -> List["QuicPacket"]:
        """Use instead of the constructor when it's possible that
        multiple quic packets could be in a single UDP packet.
        """
        data = data["_source"]["layers"]
        if not isinstance(data["quic"], list):
            return [cls(data)]

        # Create a deep copy so that we dont have any "weird" behaviour where
        # some parts of the dicts are the original and others are new
        data = copy.deepcopy(data)
        quic_src_packets = data["quic"]
        data["quic"] = None

        packets = []
        for quic_src in quic_src_packets:
            if len(quic_src.keys()) == 1 and "_ws.expert" in quic_src:
                _LOGGER.info(
                    "Discarding packet with wireshark comment %s in frame %s",
                    quic_src, data["frame"]["frame.number"])
                continue

            clone = copy.deepcopy(data)
            clone["quic"] = quic_src
            packets.append(cls(clone))

        return packets

    @property
    def timestamp(self) -> float:
        """Time since epoch (frame.time_epoch)"""
        return float(self.details["frame"]["frame.time_epoch"])

    @property
    def packet_number(self) -> int:
        """QUIC packet number (quic.packet_number)"""
        if "quic.packet_number" in self.details["quic"]:
            return int(self.details["quic"]["quic.packet_number"])
        if "quic.short" in self.details["quic"]:
            return int(
                self.details["quic"]["quic.short"]["quic.packet_number"]
            )
        if set(self.details["quic"].keys()) == {
            "quic.frame", "quic.packet_length", "quic.remaining_payload"
        }:
            # So far this seems to be the initial packet
            if self.details["frame"]["frame.number"] == "3":
                breakpoint()
            return 0
        raise NotImplementedError("Unable to get packet number")

    @property
    def packet_length(self) -> int:
        """The total length of the QUIC packet."""
        return self.details["quic"]["quic.packet_length"]

    def is_outgoing(self) -> bool:
        """Returns true iff the packet is an outgoing packet, i.e. one
        destined for port 443.
        """
        return self.details["udp"]["udp.srcport"] != "443"

    def frames(self, filter_=None) -> List[dict]:
        """Return the frames in the packet."""
        if filter_ is None:
            return self.details["quic"]["quic.frame"]
        return [frame for frame in self.frames() if filter_(frame)]

    def stream_frames(self) -> List[StreamFrame]:
        """A helper that returns stream frame objects instead of dicts."""
        return [StreamFrame(frame) for frame in self.frames(
            lambda x: "quic.stream.stream_id" in x)]

    def padding(self) -> int:
        """The total amount QUIC padding added to the packet."""
        return sum(int(frame["quic.padding_length"]) for frame in
                   self.frames(lambda f: f["quic.frame_type"] == "0"))


def parse_quic_packets(pcap_file: str, display_filter: str = "") \
        -> Iterator[QuicPacket]:
    """Parse QUIC packets from the specified pcap file"""
    command = ["tshark", "-r", pcap_file, "-T", "json", "-J", "frame udp quic",
               "--no-duplicate-keys"]
    if display_filter:
        command += ["-Y", display_filter]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    assert result.stdout is not None
    print(result.args)

    for entry in json.loads(result.stdout):
        yield from QuicPacket.from_json(entry)


class ChaffTrafficResult(NamedTuple):
    """Amount of chaff traffic present in specified packet."""
    timestamp: float
    packet_number: int
    is_outgoing: bool
    chaff_traffic: int
    other_traffic: int
    packet_length: int


def parse_chaff_traffic(pcap_file: str, chaff_streams_file: str) \
        -> List[ChaffTrafficResult]:
    """Parse only chaff traffic from the pcap_file and file specifying
    chaff streams.  Only the details of packets with chaff traffic are
    returned.
    """
    results = []
    chaff_streams = set(map(int, Path(chaff_streams_file).read_text().split()))

    display_filter = ("quic.stream.stream_id or quic.padding_length > 0 "
                      "or quic.packet_number == 0")
    for packet in parse_quic_packets(pcap_file, display_filter=display_filter):
        chaff_traffic = 0
        if packet.is_outgoing():
            # Outgoing chaff traffic is padding frames
            chaff_traffic = packet.padding()
        else:
            # Incoming cover traffic is data on specified streams
            for frame in packet.stream_frames():
                if frame.stream_id in chaff_streams:
                    chaff_traffic += frame.data_length()

        other_traffic = sum(
            frame.data_length() for frame in packet.stream_frames()
            if frame.stream_id not in chaff_streams)

        results.append(ChaffTrafficResult(
            timestamp=packet.timestamp,
            packet_number=packet.packet_number,
            is_outgoing=packet.is_outgoing(),
            chaff_traffic=chaff_traffic,
            other_traffic=other_traffic,
            packet_length=packet.packet_length,
        ))

    return results


class AllTrafficResult(NamedTuple):
    """Structured packet from trace."""
    timestamp: float
    packet_number: int
    is_outgoing: bool
    packet_length: int


def parse_all_traffic(pcap_file: str):
    """Parse the full trace from pcap_file. Details of all packets are returned
    """

    results = []

    for packet in parse_quic_packets(pcap_file):
        results.append(AllTrafficResult(
            timestamp=packet.timestamp,
            packet_number=packet.packet_number,
            is_outgoing=packet.is_outgoing(),
            packet_length=packet.packet_length,
        ))

    return results
