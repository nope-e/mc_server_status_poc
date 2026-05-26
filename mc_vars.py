from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol, Self


SEGMENT_BITS = 0x7F
CONTINUE_BIT = 0x80


class _Readable(Protocol):
    def read(self, size: int = -1) -> bytes: ...


class _Recvable(Protocol):
    def recv(self, bufsize: int) -> bytes: ...


class _Writable(Protocol):
    def write(self, data: bytes) -> int | None: ...


class _Sendable(Protocol):
    def sendall(self, data: bytes) -> None: ...


@dataclass(frozen=True, slots=True)
class _MCVarBase:
    value: int

    BIT_WIDTH: ClassVar[int]
    MAX_BYTES: ClassVar[int]

    def __post_init__(self) -> None:
        self._validate_range(self.value)

    def __int__(self) -> int:
        return self.value

    def __bytes__(self) -> bytes:
        return self.to_bytes()

    @classmethod
    def min_value(cls) -> int:
        return -(1 << (cls.BIT_WIDTH - 1))

    @classmethod
    def max_value(cls) -> int:
        return (1 << (cls.BIT_WIDTH - 1)) - 1

    @classmethod
    def _mask(cls) -> int:
        return (1 << cls.BIT_WIDTH) - 1

    @classmethod
    def _validate_range(cls, value: int) -> None:
        if not cls.min_value() <= value <= cls.max_value():
            raise ValueError(
                f"{cls.__name__} value must be between "
                f"{cls.min_value()} and {cls.max_value()}, got {value}"
            )

    @classmethod
    def _to_unsigned(cls, value: int) -> int:
        cls._validate_range(value)
        return value & cls._mask()

    @classmethod
    def _to_signed(cls, value: int) -> int:
        value &= cls._mask()
        sign_bit = 1 << (cls.BIT_WIDTH - 1)
        if value & sign_bit:
            return value - (1 << cls.BIT_WIDTH)
        return value

    @classmethod
    def encode(cls, value: int) -> bytes:
        unsigned_value = cls._to_unsigned(value)
        output = bytearray()

        while True:
            if (unsigned_value & ~SEGMENT_BITS) == 0:
                output.append(unsigned_value)
                return bytes(output)

            output.append((unsigned_value & SEGMENT_BITS) | CONTINUE_BIT)
            unsigned_value >>= 7

    @classmethod
    def decode_from(
        cls,
        data: bytes | bytearray | memoryview,
        offset: int = 0,
    ) -> tuple[int, int]:
        view = memoryview(data)
        if offset < 0 or offset > len(view):
            raise ValueError(f"offset out of range: {offset}")

        value = 0
        position = 0
        bytes_read = 0

        for index in range(offset, len(view)):
            bytes_read += 1
            current_byte = view[index]
            value = (value | ((current_byte & SEGMENT_BITS) << position)) & cls._mask()

            if (current_byte & CONTINUE_BIT) == 0:
                return cls._to_signed(value), index + 1

            if bytes_read >= cls.MAX_BYTES:
                raise ValueError(f"{cls.__name__} is too big")

            position += 7

        raise EOFError(f"Incomplete {cls.__name__} payload")

    @classmethod
    def decode(cls, data: bytes | bytearray | memoryview) -> int:
        value, next_offset = cls.decode_from(data)
        if next_offset != len(data):
            raise ValueError(
                f"{cls.__name__} decode expected exactly one value, "
                f"but {len(data) - next_offset} extra byte(s) remain"
            )
        return value

    @classmethod
    def from_bytes(
        cls,
        data: bytes | bytearray | memoryview,
        offset: int = 0,
    ) -> tuple[Self, int]:
        value, next_offset = cls.decode_from(data, offset)
        return cls(value), next_offset

    @classmethod
    def _read_one_byte(cls, source: _Readable | _Recvable) -> int:
        if hasattr(source, "recv"):
            chunk = source.recv(1)
        elif hasattr(source, "read"):
            chunk = source.read(1)
        else:
            raise TypeError("source must provide recv(1) or read(1)")

        if not chunk:
            raise EOFError(f"Unexpected EOF while reading {cls.__name__}")

        return chunk[0]

    @classmethod
    def read_value(cls, source: _Readable | _Recvable) -> int:
        value = 0
        position = 0
        bytes_read = 0

        while True:
            bytes_read += 1
            current_byte = cls._read_one_byte(source)
            value = (value | ((current_byte & SEGMENT_BITS) << position)) & cls._mask()

            if (current_byte & CONTINUE_BIT) == 0:
                return cls._to_signed(value)

            if bytes_read >= cls.MAX_BYTES:
                raise ValueError(f"{cls.__name__} is too big")

            position += 7

    @classmethod
    def read(cls, source: _Readable | _Recvable) -> Self:
        return cls(cls.read_value(source))

    @classmethod
    def write_value(cls, target: _Writable | _Sendable, value: int) -> int:
        payload = cls.encode(value)

        if hasattr(target, "sendall"):
            target.sendall(payload)
            return len(payload)

        if hasattr(target, "write"):
            written = target.write(payload)
            if written is None:
                return len(payload)
            if written != len(payload):
                raise IOError(
                    f"Short write for {cls.__name__}: expected {len(payload)} bytes, got {written}"
                )
            return written

        raise TypeError("target must provide sendall(bytes) or write(bytes)")

    @classmethod
    def size_of(cls, value: int) -> int:
        return len(cls.encode(value))

    def to_bytes(self) -> bytes:
        return self.encode(self.value)

    def write_to(self, target: _Writable | _Sendable) -> int:
        return self.write_value(target, self.value)


class VarInt(_MCVarBase):
    BIT_WIDTH = 32
    MAX_BYTES = 5


class VarLong(_MCVarBase):
    BIT_WIDTH = 64
    MAX_BYTES = 10
