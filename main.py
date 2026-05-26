import json
import os
import socket
import time

import mc_pack


def main() -> None:
    addr = (os.environ["mc_addr"], int(os.environ.get("mc_port", "25565")))
    protocol_version = int(os.environ.get("mc_protocol_version", "775"))
    timeout = float(os.environ.get("mc_timeout", "5"))

    print(f"Connecting to {addr}")

    with socket.create_connection(addr, timeout=timeout) as sock:
        handshake = mc_pack.HandShake(
            protocol_version=protocol_version,
            server_addr=addr[0],
            server_port=addr[1],
            intent=mc_pack.HandShake.STATUS,
        )
        handshake.send_to(sock)
        mc_pack.StatusRequest().send_to(sock)

        status_response = mc_pack.StatusResponse.read_from(sock)
        print("Status response:")
        print(json.dumps(status_response.data, ensure_ascii=False, indent=2))

        timestamp = time.time_ns() // 1_000_000
        ping_started = time.perf_counter()
        mc_pack.PingRequest(timestamp=timestamp).send_to(sock)
        pong_response = mc_pack.PongResponse.read_from(sock)
        latency_ms = (time.perf_counter() - ping_started) * 1000

        if pong_response.timestamp != timestamp:
            raise ValueError(
                f"Pong timestamp mismatch: expected {timestamp}, got {pong_response.timestamp}"
            )

        print(f"Pong OK, RTT: {latency_ms:.2f} ms")


if __name__ == "__main__":
    main()
