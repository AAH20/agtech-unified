"""LoRaWAN simulator: physical layer, MAC layer, and network server.

Models the LoRaWAN stack for agricultural IoT deployments:
- SpreadingFactor: SF7-SF12 with data rate and symbol rate
- LoRaRadio: air time, link budget, and sensitivity
- DutyCycleManager: EU868-style 1% duty-cycle enforcement
- DeviceSession: per-device MAC state and frame counters
- ADRState: adaptive data rate state machine
- NetworkServer: join, uplink, downlink, and history
- LoRaWANSimulator: end-to-end composition
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

# ===========================================================================
# Physical Layer
# ===========================================================================


class SpreadingFactor(Enum):
    """LoRa spreading factors SF7-SF12."""

    SF7 = 7
    SF8 = 8
    SF9 = 9
    SF10 = 10
    SF11 = 11
    SF12 = 12

    @property
    def data_rate_bps(self) -> float:
        """Data rate in bits per second (BW=125kHz, CR=4/5)."""
        return self.value * 125_000 / (2**self.value) * 0.8

    @property
    def symbol_rate(self) -> float:
        """Symbol rate in symbols per second."""
        return 125_000 / (2**self.value)


class LoRaRadio:
    """LoRa physical-layer model: air time, link budget, sensitivity."""

    PREAMBLE_SYMBOLS = 8
    PREAMBLE_OVERHEAD = 4.25  # LoRa preamble overhead symbols
    CODING_RATE_DENOM = 5  # CR=4/5
    FREQUENCY_MHZ = 868.0  # EU868 band
    NOISE_FIGURE_DB = 6.0

    # Minimum SNR required per SF (dB)
    SNR_REQUIRED_DB = {
        SpreadingFactor.SF7: -7.5,
        SpreadingFactor.SF8: -10.0,
        SpreadingFactor.SF9: -12.5,
        SpreadingFactor.SF10: -15.0,
        SpreadingFactor.SF11: -17.5,
        SpreadingFactor.SF12: -20.0,
    }

    def air_time(self, payload_bytes: int, sf: SpreadingFactor) -> float:
        """Total on-air time in seconds for a frame."""
        symbol_rate = sf.symbol_rate
        preamble_time = (self.PREAMBLE_SYMBOLS + self.PREAMBLE_OVERHEAD) / symbol_rate

        # Payload symbol count (LoRa spec)
        pl = payload_bytes
        sfr = sf.value
        numerator = 8 * pl - 4 * sfr + 28 + 16
        denominator = 4 * sfr
        payload_symbols = 8 + max(math.ceil(numerator / denominator), 0) * (
            self.CODING_RATE_DENOM + 4
        )
        payload_time = payload_symbols / symbol_rate
        return preamble_time + payload_time

    def received_power_dbm(self, distance_m: float, tx_power_dbm: float) -> float:
        """Received power via free-space path loss."""
        if distance_m <= 0:
            return tx_power_dbm
        # FSPL = 20*log10(d_km) + 20*log10(f_MHz) + 32.44
        fspl = 20 * math.log10(distance_m / 1000.0) + 20 * math.log10(self.FREQUENCY_MHZ) + 32.44
        return tx_power_dbm - fspl

    def sensitivity_dbm(self, sf: SpreadingFactor) -> float:
        """Receiver sensitivity in dBm for a given SF."""
        # sensitivity = -174 + 10*log10(BW) + NF + SNR_required
        bw_hz = 125_000
        return -174.0 + 10 * math.log10(bw_hz) + self.NOISE_FIGURE_DB + self.SNR_REQUIRED_DB[sf]

    def max_distance_m(self, sf: SpreadingFactor, tx_power_dbm: float) -> float:
        """Maximum usable distance where TX power meets sensitivity."""
        max_path_loss = tx_power_dbm - self.sensitivity_dbm(sf)
        # Invert FSPL: d_km = 10^((FSPL - 32.44 - 20*log10(f_MHz)) / 20)
        d_km = 10 ** ((max_path_loss - 32.44 - 20 * math.log10(self.FREQUENCY_MHZ)) / 20)
        return d_km * 1000.0


# ===========================================================================
# MAC Layer: Duty Cycle
# ===========================================================================


class DutyCycleManager:
    """Sliding-window duty-cycle enforcement (EU868 1% default).

    The EU868 band enforces a 1% duty cycle over a 1-hour window,
    giving 36 seconds of air time per hour.
    """

    def __init__(self, duty_cycle: float = 0.01, window_seconds: float = 3600.0) -> None:
        if not 0 < duty_cycle <= 1:
            raise ValueError("duty_cycle must be in (0, 1]")
        self.duty_cycle = duty_cycle
        self.window_seconds = window_seconds
        self._transmissions: Deque[Tuple[float, float]] = deque()  # (start, air_time)

    def _current_usage(self, now: float) -> float:
        """Total air time within the sliding window."""
        cutoff = now - self.window_seconds
        while self._transmissions and self._transmissions[0][0] < cutoff:
            self._transmissions.popleft()
        return sum(air for _, air in self._transmissions)

    def can_transmit(self, now: float, air_time: float) -> bool:
        """True if the transmission fits within the duty-cycle budget."""
        return self._current_usage(now) + air_time <= self.duty_cycle * self.window_seconds

    def record_transmission(self, now: float, air_time: float) -> None:
        """Record a transmission for duty-cycle accounting."""
        self._transmissions.append((now, air_time))


# ===========================================================================
# MAC Layer: Device Session
# ===========================================================================


@dataclass
class DeviceSession:
    """Per-device MAC state: keys, frame counters, replay protection."""

    dev_addr: str
    app_key: Optional[str] = None
    joined: bool = False
    fcnt_up: int = 0
    fcnt_down: int = 0
    _seen_uplink_fcounters: Set[int] = field(default_factory=set)

    def join(self, app_key: str) -> None:
        """Activate the session with an application key."""
        self.app_key = app_key
        self.joined = True
        self.fcnt_up = 0
        self.fcnt_down = 0
        self._seen_uplink_fcounters.clear()

    def record_uplink(self, fcnt: int) -> None:
        """Record an uplink frame counter."""
        self.fcnt_up = max(self.fcnt_up, fcnt)
        self._seen_uplink_fcounters.add(fcnt)

    def is_replay(self, fcnt: int) -> bool:
        """True if this frame counter has been seen before."""
        return fcnt in self._seen_uplink_fcounters


# ===========================================================================
# MAC Layer: Adaptive Data Rate
# ===========================================================================


class ADRState:
    """Adaptive data rate state machine.

    Steps SF down (faster) when SNR exceeds the margin threshold,
    steps SF up (more robust) when SNR falls below it.
    """

    MARGIN_THRESHOLD_DB = 5.0

    def __init__(self) -> None:
        self._sf = SpreadingFactor.SF12

    @property
    def sf(self) -> SpreadingFactor:
        return self._sf

    def update(self, snr_db: float, margin_db: float) -> None:
        """Update SF based on observed SNR and the configured margin."""
        if snr_db > margin_db + self.MARGIN_THRESHOLD_DB:
            self._step_down()
        elif snr_db < margin_db - self.MARGIN_THRESHOLD_DB:
            self._step_up()

    def _step_down(self) -> None:
        """Move one step toward SF7 (faster)."""
        if self._sf != SpreadingFactor.SF7:
            self._sf = SpreadingFactor(self._sf.value - 1)

    def _step_up(self) -> None:
        """Move one step toward SF12 (more robust)."""
        if self._sf != SpreadingFactor.SF12:
            self._sf = SpreadingFactor(self._sf.value + 1)


# ===========================================================================
# Network Server
# ===========================================================================


class NetworkServer:
    """LoRaWAN network server: join, uplink, downlink, history."""

    def __init__(self) -> None:
        self._sessions: Dict[str, DeviceSession] = {}
        self._uplink_history: Dict[str, List[Dict[str, Any]]] = {}
        self._downlink_queues: Dict[str, Deque[Dict[str, Any]]] = {}

    def join_accept(self, dev_eui: str, dev_addr: str) -> DeviceSession:
        """Accept a join request and create a device session."""
        session = DeviceSession(dev_addr=dev_addr)
        session.join(app_key=f"key-{dev_eui}")
        self._sessions[dev_addr] = session
        self._uplink_history[dev_addr] = []
        self._downlink_queues[dev_addr] = deque()
        return session

    def is_joined(self, dev_addr: str) -> bool:
        """True if the device has an active session."""
        return dev_addr in self._sessions

    def handle_uplink(
        self,
        dev_addr: str,
        fcnt: int,
        payload: bytes,
        rssi: float,
        snr: float,
    ) -> Dict[str, Any]:
        """Process an uplink frame. Returns {accepted, reason?}."""
        if not self.is_joined(dev_addr):
            return {"accepted": False, "reason": "not_joined"}
        session = self._sessions[dev_addr]
        if session.is_replay(fcnt):
            return {"accepted": False, "reason": "replay"}
        session.record_uplink(fcnt)
        self._uplink_history[dev_addr].append(
            {"fcnt": fcnt, "payload": payload, "rssi": rssi, "snr": snr}
        )
        return {"accepted": True}

    def queue_downlink(self, dev_addr: str, payload: bytes, port: int) -> None:
        """Queue a downlink frame for a device."""
        if dev_addr not in self._downlink_queues:
            self._downlink_queues[dev_addr] = deque()
        self._downlink_queues[dev_addr].append({"payload": payload, "port": port})

    def get_downlink(self, dev_addr: str) -> Optional[Dict[str, Any]]:
        """Pop the next queued downlink for a device, or None."""
        queue = self._downlink_queues.get(dev_addr)
        if not queue:
            return None
        return queue.popleft()

    def get_uplink_history(self, dev_addr: str) -> List[Dict[str, Any]]:
        """Return the uplink history for a device."""
        return list(self._uplink_history.get(dev_addr, []))


# ===========================================================================
# End-to-End Simulator
# ===========================================================================


class LoRaWANSimulator:
    """Composes radio, duty cycle, ADR, and network server."""

    def __init__(self, duty_cycle: float = 0.01) -> None:
        self.radio = LoRaRadio()
        self.duty_cycle_mgr = DutyCycleManager(duty_cycle=duty_cycle)
        self.server = NetworkServer()
        self._adr: Dict[str, ADRState] = {}
        self._now: float = 0.0

    def join_device(self, dev_eui: str, dev_addr: str) -> None:
        """Join a device to the network."""
        self.server.join_accept(dev_eui=dev_eui, dev_addr=dev_addr)
        self._adr[dev_addr] = ADRState()

    def send_uplink(self, dev_addr: str, payload: bytes) -> Dict[str, Any]:
        """Send an uplink through the full stack."""
        if not self.server.is_joined(dev_addr):
            return {"accepted": False, "reason": "not_joined"}

        sf = self._adr[dev_addr].sf
        air_time = self.radio.air_time(payload_bytes=len(payload), sf=sf)

        if not self.duty_cycle_mgr.can_transmit(now=self._now, air_time=air_time):
            return {"accepted": False, "reason": "duty_cycle"}

        self.duty_cycle_mgr.record_transmission(now=self._now, air_time=air_time)
        self._now += air_time

        # Simulate a reasonable SNR for the ADR state machine
        snr = 10.0 if sf.value <= 9 else 5.0
        result = self.server.handle_uplink(
            dev_addr=dev_addr,
            fcnt=self.server._sessions[dev_addr].fcnt_up + 1,
            payload=payload,
            rssi=-80.0,
            snr=snr,
        )
        if result["accepted"]:
            self._adr[dev_addr].update(snr_db=snr, margin_db=10.0)
        return result

    def queue_downlink(self, dev_addr: str, payload: bytes, port: int) -> None:
        """Queue a downlink for a device."""
        self.server.queue_downlink(dev_addr=dev_addr, payload=payload, port=port)

    def receive_downlink(self, dev_addr: str) -> Optional[Dict[str, Any]]:
        """Receive a queued downlink."""
        return self.server.get_downlink(dev_addr=dev_addr)

    def get_sf(self, dev_addr: str) -> SpreadingFactor:
        """Get the current SF for a device."""
        return self._adr[dev_addr].sf
