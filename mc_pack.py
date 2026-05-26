from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

from mc_vars import VarInt


class _Recvable(Protocol):
    def recv(self, bufsize: int) -> bytes: ...


class _Sendable(Protocol):
    def sendall(self, data: bytes) -> None: ...


def encode_mc_string(value: str, max_chars: int) -> bytes:
    if len(value) > max_chars:
        raise ValueError(f"String is too long: {len(value)} > {max_chars}")

    encoded = value.encode("utf-8")
    return VarInt.encode(len(encoded)) + encoded


def decode_mc_string(
    data: bytes | bytearray | memoryview,
    offset: int = 0,
    max_chars: int = 32767,
) -> tuple[str, int]:
    byte_length, offset = VarInt.decode_from(data, offset)
    if byte_length < 0:
        raise ValueError("String byte length cannot be negative")

    end = offset + byte_length
    view = memoryview(data)
    if end > len(view):
        raise EOFError("Incomplete Minecraft string payload")

    value = bytes(view[offset:end]).decode("utf-8")
    if len(value) > max_chars:
        raise ValueError(f"Decoded string is too long: {len(value)} > {max_chars}")

    return value, end


def read_exact(source: _Recvable, size: int) -> bytes:
    chunks = bytearray()

    while len(chunks) < size:
        chunk = source.recv(size - len(chunks))
        if not chunk:
            raise EOFError(f"Unexpected EOF while reading {size} bytes")
        chunks.extend(chunk)

    return bytes(chunks)


def read_packet(source: _Recvable) -> tuple[int, bytes]:
    packet_length = VarInt.read_value(source)
    packet_data = read_exact(source, packet_length)
    packet_id, offset = VarInt.decode_from(packet_data)
    return packet_id, packet_data[offset:]


class BaseMCPacket:
    packet_id: int = 0

    def payload_bytes(self) -> bytes:
        return b""

    def packet_bytes(self) -> bytes:
        return VarInt.encode(self.packet_id) + self.payload_bytes()

    def packet_length(self) -> int:
        return len(self.packet_bytes())

    def encoded_length_bytes(self) -> bytes:
        return VarInt.encode(self.packet_length())

    def serialize(self) -> bytes:
        packet = self.packet_bytes()
        return VarInt.encode(len(packet)) + packet

    def send_to(self, target: _Sendable) -> int:
        payload = self.serialize()
        target.sendall(payload)
        return len(payload)

    def __bytes__(self) -> bytes:
        return self.serialize()

    def __len__(self) -> int:
        return self.packet_length()


@dataclass(slots=True)
class HandShake(BaseMCPacket):
    STATUS: ClassVar[int] = 1
    LOGIN: ClassVar[int] = 2
    TRANSFER: ClassVar[int] = 3

    protocol_version: int = 775
    server_addr: str = "localhost"
    server_port: int = 25565
    intent: int = STATUS
    packet_id: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if len(self.server_addr) > 255:
            raise ValueError("server_addr must be 255 characters or fewer")

        if not 0 <= self.server_port <= 0xFFFF:
            raise ValueError("server_port must be between 0 and 65535")

        if self.intent not in (self.STATUS, self.LOGIN, self.TRANSFER):
            raise ValueError("intent must be 1 (status), 2 (login), or 3 (transfer)")

    def payload_bytes(self) -> bytes:
        return b"".join(
            (
                VarInt.encode(self.protocol_version),
                encode_mc_string(self.server_addr, 255),
                struct.pack(">H", self.server_port),
                VarInt.encode(self.intent),
            )
        )


@dataclass(slots=True)
class StatusRequest(BaseMCPacket):
    packet_id: int = field(init=False, default=0)


@dataclass(slots=True)
class PingRequest(BaseMCPacket):
    timestamp: int
    packet_id: int = field(init=False, default=1)

    def __post_init__(self) -> None:
        if not -(1 << 63) <= self.timestamp <= (1 << 63) - 1:
            raise ValueError("timestamp must fit in a signed 64-bit integer")

    def payload_bytes(self) -> bytes:
        return struct.pack(">q", self.timestamp)


@dataclass(slots=True)
class StatusResponse:
    packet_id: ClassVar[int] = 0

    json_response: str

    @property
    def data(self) -> Any:
        return json.loads(self.json_response)

    @classmethod
    def from_payload(cls, payload: bytes) -> StatusResponse:
        json_response, offset = decode_mc_string(payload, max_chars=32767)
        if offset != len(payload):
            raise ValueError(
                f"StatusResponse has {len(payload) - offset} unexpected trailing byte(s)"
            )
        return cls(json_response=json_response)

    @classmethod
    def read_from(cls, source: _Recvable) -> StatusResponse:
        packet_id, payload = read_packet(source)
        if packet_id != cls.packet_id:
            raise ValueError(f"Expected packet id {cls.packet_id}, got {packet_id}")
        return cls.from_payload(payload)


@dataclass(slots=True)
class PongResponse:
    packet_id: ClassVar[int] = 1

    timestamp: int

    @classmethod
    def from_payload(cls, payload: bytes) -> PongResponse:
        if len(payload) != 8:
            raise ValueError(f"PongResponse payload must be 8 bytes, got {len(payload)}")
        return cls(timestamp=struct.unpack(">q", payload)[0])

    @classmethod
    def read_from(cls, source: _Recvable) -> PongResponse:
        packet_id, payload = read_packet(source)
        if packet_id != cls.packet_id:
            raise ValueError(f"Expected packet id {cls.packet_id}, got {packet_id}")
        return cls.from_payload(payload)
