"""BLE link to the car's HM-10 UART module. The firmware stops the motors after 500 ms without a command."""

import asyncio
import os
import time
from collections import deque

from bleak import BleakClient, BleakScanner

from airo.protocol import Sens, cmd_packet, parse_sens

UART_CHAR = "0000ffe1-0000-1000-8000-00805f9b34fb"
DEFAULT_NAME = "HC-05"


class Car:
    def __init__(self, client: BleakClient):
        self._client = client
        self._buf = bytearray()
        self._write: asyncio.Task | None = None
        self.sens: Sens | None = None
        self.sens_count = 0
        self.ack_ms: deque[float] = deque(maxlen=200)

    @classmethod
    async def connect(cls, address: str | None = None, timeout: float = 20) -> "Car":
        address = address or os.getenv("AIRO_BLE_ADDRESS")
        if not address:
            dev = await BleakScanner.find_device_by_name(DEFAULT_NAME, timeout=timeout)
            if dev is None:
                raise RuntimeError(f"no BLE device named {DEFAULT_NAME}")
            address = dev.address
        client = BleakClient(address, timeout=timeout)
        await client.connect()
        car = cls(client)
        await client.start_notify(UART_CHAR, car._on_notify)
        return car

    def _on_notify(self, _char, data: bytearray):
        self._buf.extend(data)
        for s in parse_sens(self._buf):
            self.sens = s
            self.sens_count += 1

    def drive(self, pwm_l: int, pwm_r: int) -> bool:
        """Send without blocking; returns False if the previous write is still in flight."""
        if self._write and not self._write.done():
            return False
        self._write = asyncio.create_task(self._send(cmd_packet(pwm_l, pwm_r)))
        return True

    async def _send(self, pkt: bytes):
        t0 = time.perf_counter()
        # FFE1 silently drops write-without-response.
        await self._client.write_gatt_char(UART_CHAR, pkt, response=True)
        self.ack_ms.append((time.perf_counter() - t0) * 1000)

    async def stop(self):
        if self._write:
            await asyncio.gather(self._write, return_exceptions=True)
        await self._send(cmd_packet(0, 0))

    async def close(self):
        try:
            await self.stop()
        finally:
            await self._client.disconnect()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()
