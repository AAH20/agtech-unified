"""Tests for the LoRaWAN simulator (IOT gap: no LoRaWAN simulator).

Covers the physical-layer model (spreading factor <-> data rate, air time),
the MAC layer (duty-cycle enforcement, ADR), and the network-server side
(device session, uplink/downlink, receive windows).
"""

import math

import pytest

from src.iot.lorawan import (
    ADRState,
    DeviceSession,
    DutyCycleManager,
    LoRaRadio,
    LoRaWANSimulator,
    NetworkServer,
    SpreadingFactor,
)


class TestSpreadingFactor:
    """Spreading factor <-> data rate mapping."""

    def test_sf7_fastest(self):
        """SF7 has the highest data rate (shortest on-air time)."""
        assert SpreadingFactor.SF7.data_rate_bps > SpreadingFactor.SF12.data_rate_bps

    def test_sf12_slowest(self):
        """SF12 has the lowest data rate."""
        rates = [sf.data_rate_bps for sf in SpreadingFactor]
        assert rates == sorted(rates, reverse=True)

    def test_symbol_rate(self):
        """Symbol rate is bandwidth / 2^SF."""
        assert SpreadingFactor.SF7.symbol_rate == pytest.approx(125000 / 128)
        assert SpreadingFactor.SF12.symbol_rate == pytest.approx(125000 / 4096)


class TestLoRaRadio:
    """Physical-layer air time and link budget."""

    def test_air_time_increases_with_payload(self):
        """Larger payloads take longer on air at the same SF."""
        radio = LoRaRadio()
        t1 = radio.air_time(payload_bytes=10, sf=SpreadingFactor.SF7)
        t2 = radio.air_time(payload_bytes=50, sf=SpreadingFactor.SF7)
        assert t2 > t1

    def test_air_time_increases_with_sf(self):
        """Higher SF (slower) means longer air time for the same payload."""
        radio = LoRaRadio()
        t7 = radio.air_time(payload_bytes=10, sf=SpreadingFactor.SF7)
        t12 = radio.air_time(payload_bytes=10, sf=SpreadingFactor.SF12)
        assert t12 > t7

    def test_air_time_zero_payload(self):
        """A zero-payload frame still has a preamble cost."""
        radio = LoRaRadio()
        assert radio.air_time(payload_bytes=0, sf=SpreadingFactor.SF7) > 0.0

    def test_link_budget(self):
        """Received power falls with distance (FSPL)."""
        radio = LoRaRadio()
        near = radio.received_power_dbm(distance_m=100.0, tx_power_dbm=14.0)
        far = radio.received_power_dbm(distance_m=1000.0, tx_power_dbm=14.0)
        assert near > far
        # Doubling distance costs ~6 dB
        d1 = radio.received_power_dbm(distance_m=100.0, tx_power_dbm=14.0)
        d2 = radio.received_power_dbm(distance_m=200.0, tx_power_dbm=14.0)
        assert d1 - d2 == pytest.approx(20 * math.log10(2), abs=0.1)

    def test_sensitivity_improves_with_sf(self):
        """Higher SF reaches further (lower sensitivity)."""
        radio = LoRaRadio()
        sens7 = radio.sensitivity_dbm(SpreadingFactor.SF7)
        sens12 = radio.sensitivity_dbm(SpreadingFactor.SF12)
        assert sens12 < sens7

    def test_max_distance(self):
        """Max usable distance is where TX power meets sensitivity."""
        radio = LoRaRadio()
        d7 = radio.max_distance_m(sf=SpreadingFactor.SF7, tx_power_dbm=14.0)
        d12 = radio.max_distance_m(sf=SpreadingFactor.SF12, tx_power_dbm=14.0)
        assert d12 > d7 > 0.0


class TestDutyCycleManager:
    """EU868-style 1% duty-cycle enforcement."""

    def test_allows_first_transmission(self):
        """A fresh manager allows a transmission."""
        mgr = DutyCycleManager(duty_cycle=0.01)
        assert mgr.can_transmit(now=0.0, air_time=0.1) is True

    def test_blocks_when_budget_exhausted(self):
        """Transmission is blocked when the sub-band budget is exhausted."""
        mgr = DutyCycleManager(duty_cycle=0.01)
        # Consume the whole 1% budget (36s in a 3600s window)
        mgr.record_transmission(now=0.0, air_time=36.0)
        assert mgr.can_transmit(now=60.0, air_time=0.1) is False

    def test_allows_after_window_passes(self):
        """Budget frees up once the averaging window has passed."""
        mgr = DutyCycleManager(duty_cycle=0.01)
        mgr.record_transmission(now=0.0, air_time=36.0)
        assert mgr.can_transmit(now=3601.0, air_time=0.1) is True

    def test_partial_budget(self):
        """A transmission fitting the remaining budget is allowed."""
        mgr = DutyCycleManager(duty_cycle=0.01)
        mgr.record_transmission(now=0.0, air_time=35.0)
        assert mgr.can_transmit(now=60.0, air_time=0.5) is True
        assert mgr.can_transmit(now=60.0, air_time=1.5) is False

    def test_invalid_duty_cycle(self):
        """Duty cycle must be in (0, 1]."""
        with pytest.raises(ValueError):
            DutyCycleManager(duty_cycle=0.0)
        with pytest.raises(ValueError):
            DutyCycleManager(duty_cycle=1.5)


class TestDeviceSession:
    """Per-device MAC state and frame counters."""

    def test_join_sets_session(self):
        """Joining assigns keys and resets counters."""
        session = DeviceSession(dev_addr="00:11:22:33")
        session.join(app_key="0123456789abcdef0123456789abcdef")
        assert session.joined is True
        assert session.fcnt_up == 0
        assert session.fcnt_down == 0

    def test_uplink_increments_fcnt(self):
        """Each uplink increments the up frame counter."""
        session = DeviceSession(dev_addr="00:11:22:33")
        session.join(app_key="0123456789abcdef0123456789abcdef")
        session.record_uplink(fcnt=1)
        session.record_uplink(fcnt=2)
        assert session.fcnt_up == 2

    def test_replay_detection(self):
        """A repeated frame counter is flagged as replay."""
        session = DeviceSession(dev_addr="00:11:22:33")
        session.join(app_key="0123456789abcdef0123456789abcdef")
        session.record_uplink(fcnt=5)
        assert session.is_replay(fcnt=5) is True
        assert session.is_replay(fcnt=6) is False

    def test_not_joined(self):
        """A fresh session is not joined."""
        session = DeviceSession(dev_addr="00:11:22:33")
        assert session.joined is False


class TestADRState:
    """Adaptive data rate state machine."""

    def test_initial_sf(self):
        """ADR starts at the slowest SF."""
        adr = ADRState()
        assert adr.sf == SpreadingFactor.SF12

    def test_improve_sf_on_good_margin(self):
        """Sufficient margin steps the SF down (faster)."""
        adr = ADRState()
        adr.update(snr_db=20.0, margin_db=10.0)
        assert adr.sf == SpreadingFactor.SF11

    def test_worsen_sf_on_poor_margin(self):
        """Poor margin steps the SF up (more robust)."""
        adr = ADRState()
        adr.update(snr_db=-15.0, margin_db=10.0)
        assert adr.sf == SpreadingFactor.SF12

    def test_sf_bounds(self):
        """SF never goes below SF7 or above SF12."""
        adr = ADRState()
        for _ in range(20):
            adr.update(snr_db=30.0, margin_db=10.0)
        assert adr.sf == SpreadingFactor.SF7
        for _ in range(20):
            adr.update(snr_db=-30.0, margin_db=10.0)
        assert adr.sf == SpreadingFactor.SF12


class TestNetworkServer:
    """Network server: join, uplink, downlink scheduling."""

    def test_join_accept(self):
        """A join-accept registers the device session."""
        ns = NetworkServer()
        ns.join_accept(dev_eui="AA:BB", dev_addr="00:11:22:33")
        assert ns.is_joined("00:11:22:33") is True

    def test_uplink_accepted(self):
        """An uplink from a joined device is accepted."""
        ns = NetworkServer()
        ns.join_accept(dev_eui="AA:BB", dev_addr="00:11:22:33")
        result = ns.handle_uplink(
            dev_addr="00:11:22:33", fcnt=1, payload=b"\x01\x02", rssi=-80.0, snr=8.0
        )
        assert result["accepted"] is True

    def test_uplink_rejected_when_not_joined(self):
        """An uplink from an unknown device is rejected."""
        ns = NetworkServer()
        result = ns.handle_uplink(dev_addr="FF:FF", fcnt=1, payload=b"\x01", rssi=-80.0, snr=8.0)
        assert result["accepted"] is False

    def test_uplink_replay_rejected(self):
        """A replayed frame counter is rejected."""
        ns = NetworkServer()
        ns.join_accept(dev_eui="AA:BB", dev_addr="00:11:22:33")
        ns.handle_uplink(dev_addr="00:11:22:33", fcnt=5, payload=b"\x01", rssi=-80.0, snr=8.0)
        result = ns.handle_uplink(
            dev_addr="00:11:22:33", fcnt=5, payload=b"\x01", rssi=-80.0, snr=8.0
        )
        assert result["accepted"] is False

    def test_downlink_queue(self):
        """A queued downlink is returned for the device."""
        ns = NetworkServer()
        ns.join_accept(dev_eui="AA:BB", dev_addr="00:11:22:33")
        ns.queue_downlink(dev_addr="00:11:22:33", payload=b"\x02", port=2)
        downlink = ns.get_downlink(dev_addr="00:11:22:33")
        assert downlink is not None
        assert downlink["payload"] == b"\x02"
        assert downlink["port"] == 2

    def test_no_downlink_returns_none(self):
        """No queued downlink returns None."""
        ns = NetworkServer()
        ns.join_accept(dev_eui="AA:BB", dev_addr="00:11:22:33")
        assert ns.get_downlink(dev_addr="00:11:22:33") is None

    def test_uplink_history(self):
        """Uplink history is recorded per device."""
        ns = NetworkServer()
        ns.join_accept(dev_eui="AA:BB", dev_addr="00:11:22:33")
        ns.handle_uplink(dev_addr="00:11:22:33", fcnt=1, payload=b"\x01", rssi=-80.0, snr=8.0)
        ns.handle_uplink(dev_addr="00:11:22:33", fcnt=2, payload=b"\x02", rssi=-81.0, snr=7.5)
        history = ns.get_uplink_history("00:11:22:33")
        assert len(history) == 2
        assert history[0]["fcnt"] == 1
        assert history[1]["fcnt"] == 2


class TestLoRaWANSimulator:
    """End-to-end simulator: device -> radio -> duty cycle -> server."""

    def test_simulator_join_and_uplink(self):
        """A device can join and send an uplink through the simulator."""
        sim = LoRaWANSimulator(duty_cycle=1.0)
        sim.join_device(dev_eui="AA:BB", dev_addr="00:11:22:33")
        result = sim.send_uplink(dev_addr="00:11:22:33", payload=b"\x01\x02")
        assert result["accepted"] is True

    def test_simulator_duty_cycle_blocks(self):
        """Duty-cycle enforcement blocks a second immediate TX."""
        sim = LoRaWANSimulator(duty_cycle=0.00001)
        sim.join_device(dev_eui="AA:BB", dev_addr="00:11:22:33")
        sim.send_uplink(dev_addr="00:11:22:33", payload=b"\x01")
        result = sim.send_uplink(dev_addr="00:11:22:33", payload=b"\x02")
        assert result["accepted"] is False
        assert "duty_cycle" in result["reason"]

    def test_simulator_downlink_round_trip(self):
        """A downlink queued at the server reaches the device."""
        sim = LoRaWANSimulator(duty_cycle=1.0)
        sim.join_device(dev_eui="AA:BB", dev_addr="00:11:22:33")
        sim.queue_downlink(dev_addr="00:11:22:33", payload=b"\x03", port=1)
        downlink = sim.receive_downlink(dev_addr="00:11:22:33")
        assert downlink is not None
        assert downlink["payload"] == b"\x03"

    def test_simulator_adr_stable_sf(self):
        """ADR keeps the SF stable when signal is moderate."""
        sim = LoRaWANSimulator(duty_cycle=1.0)
        sim.join_device(dev_eui="AA:BB", dev_addr="00:11:22:33")
        initial_sf = sim.get_sf("00:11:22:33")
        for _ in range(5):
            sim.send_uplink(dev_addr="00:11:22:33", payload=b"\x01")
        assert sim.get_sf("00:11:22:33").value == initial_sf.value

    def test_simulator_unknown_device(self):
        """Uplink from an unknown device is rejected."""
        sim = LoRaWANSimulator()
        result = sim.send_uplink(dev_addr="FF:FF", payload=b"\x01")
        assert result["accepted"] is False
