"""Tests for solar geometry calculations in FAO-56."""

import math

import pytest

from src.digital_twin.water_balance import SolarGeometry


class TestSolarDeclination:
    """Test solar declination calculation."""

    def test_declination_summer_solstice(self):
        """Declination is ~23.45° at summer solstice (DOY 172)."""
        dec = SolarGeometry.solar_declination(172)
        assert dec == pytest.approx(math.radians(23.45), abs=0.01)

    def test_declination_winter_solstice(self):
        """Declination is ~-23.45° at winter solstice (DOY 355)."""
        dec = SolarGeometry.solar_declination(355)
        assert dec == pytest.approx(math.radians(-23.45), abs=0.01)

    def test_declination_equinox(self):
        """Declination is ~0° at equinoxes (DOY 80, 265)."""
        dec_spring = SolarGeometry.solar_declination(80)
        dec_fall = SolarGeometry.solar_declination(265)
        assert dec_spring == pytest.approx(0.0, abs=0.02)
        assert dec_fall == pytest.approx(0.0, abs=0.02)

    def test_declination_range(self):
        """Declination stays within ±23.45°."""
        for doy in range(1, 366):
            dec = SolarGeometry.solar_declination(doy)
            assert abs(dec) <= math.radians(23.45) + 1e-9


class TestSunsetHourAngle:
    """Test sunset hour angle calculation."""

    def test_equator_equinox(self):
        """At equator on equinox, sunset hour angle is 90°."""
        lat = math.radians(0.0)
        dec = SolarGeometry.solar_declination(80)  # ~0
        omega = SolarGeometry.sunset_hour_angle(lat, dec)
        assert omega == pytest.approx(math.radians(90.0), abs=0.01)

    def test_mid_latitude_summer(self):
        """Summer sunset hour angle > 90° at mid-latitudes."""
        lat = math.radians(40.0)
        dec = SolarGeometry.solar_declination(172)  # summer solstice
        omega = SolarGeometry.sunset_hour_angle(lat, dec)
        assert omega > math.radians(90.0)

    def test_mid_latitude_winter(self):
        """Winter sunset hour angle < 90° at mid-latitudes."""
        lat = math.radians(40.0)
        dec = SolarGeometry.solar_declination(355)  # winter solstice
        omega = SolarGeometry.sunset_hour_angle(lat, dec)
        assert omega < math.radians(90.0)

    def test_polar_day(self):
        """Polar day returns π (24h daylight)."""
        lat = math.radians(70.0)
        dec = math.radians(23.45)  # summer solstice
        omega = SolarGeometry.sunset_hour_angle(lat, dec)
        assert omega == pytest.approx(math.pi)


class TestExtraterrestrialRadiation:
    """Test extraterrestrial radiation (Ra) calculation."""

    def test_ra_positive(self):
        """Extraterrestrial radiation is always positive."""
        lat = math.radians(35.0)
        for doy in [1, 80, 172, 265, 355]:
            ra = SolarGeometry.extraterrestrial_radiation(lat, doy)
            assert ra > 0.0

    def test_ra_summer_higher_than_winter(self):
        """Ra is higher in summer than winter at mid-latitudes."""
        lat = math.radians(40.0)
        ra_summer = SolarGeometry.extraterrestrial_radiation(lat, 172)
        ra_winter = SolarGeometry.extraterrestrial_radiation(lat, 355)
        assert ra_summer > ra_winter

    def test_ra_units(self):
        """Ra is in MJ/m²/day (typical range 20-45)."""
        lat = math.radians(35.0)
        ra = SolarGeometry.extraterrestrial_radiation(lat, 172)
        assert 20.0 < ra < 45.0

    def test_ra_symmetric_hemispheres(self):
        """Ra is symmetric for opposite latitudes on same DOY."""
        ra_north = SolarGeometry.extraterrestrial_radiation(math.radians(35.0), 172)
        ra_south = SolarGeometry.extraterrestrial_radiation(math.radians(-35.0), 355)
        assert ra_north == pytest.approx(ra_south, rel=0.01)


class TestDaylightHours:
    """Test daylight hours calculation."""

    def test_equator_12_hours(self):
        """Equator has ~12 hours daylight year-round."""
        lat = math.radians(0.0)
        for doy in [80, 172, 265, 355]:
            hours = SolarGeometry.daylight_hours(lat, doy)
            assert hours == pytest.approx(12.0, abs=0.5)

    def test_mid_latitude_summer_longer(self):
        """Summer has longer days than winter at mid-latitudes."""
        lat = math.radians(40.0)
        summer = SolarGeometry.daylight_hours(lat, 172)
        winter = SolarGeometry.daylight_hours(lat, 355)
        assert summer > 12.0
        assert winter < 12.0

    def test_daylight_hours_range(self):
        """Daylight hours are between 0 and 24."""
        lat = math.radians(50.0)
        for doy in range(1, 366, 30):
            hours = SolarGeometry.daylight_hours(lat, doy)
            assert 0.0 <= hours <= 24.0


class TestSolarRadiationFromSunshine:
    """Test solar radiation estimation from sunshine hours."""

    def test_clear_sky_radiation(self):
        """Clear sky (max sunshine) gives higher radiation."""
        lat = math.radians(35.0)
        doy = 172
        dec = SolarGeometry.solar_declination(doy)
        omega = SolarGeometry.sunset_hour_angle(lat, dec)
        ra = SolarGeometry.extraterrestrial_radiation(lat, doy)
        # Full sunshine
        rs_clear = SolarGeometry.solar_radiation_from_sunshine(ra, omega, 1.0)
        # Half sunshine
        rs_half = SolarGeometry.solar_radiation_from_sunshine(ra, omega, 0.5)
        assert rs_clear > rs_half

    def test_radiation_proportional_to_sunshine(self):
        """Radiation scales with sunshine fraction."""
        lat = math.radians(35.0)
        doy = 172
        dec = SolarGeometry.solar_declination(doy)
        omega = SolarGeometry.sunset_hour_angle(lat, dec)
        ra = SolarGeometry.extraterrestrial_radiation(lat, doy)
        rs_0 = SolarGeometry.solar_radiation_from_sunshine(ra, omega, 0.0)
        rs_1 = SolarGeometry.solar_radiation_from_sunshine(ra, omega, 1.0)
        assert rs_0 < rs_1
        assert rs_0 >= 0.0


class TestSolarGeometryIntegration:
    """Test solar geometry integration with FAO-56."""

    def test_et0_uses_solar_geometry(self):
        """ET0 computation uses solar geometry for Ra."""
        from src.digital_twin.water_balance import WaterBalanceModel

        wb = WaterBalanceModel()
        weather = {
            "temp_max": 30.0,
            "temp_min": 20.0,
            "humidity_max": 80.0,
            "humidity_min": 40.0,
            "wind_speed": 2.0,
            "solar_radiation": 22.0,
            "latitude": 35.0,
            "day_of_year": 180,
        }
        et0 = wb.compute_et0(weather)
        assert et0 > 0.0
        assert et0 < 15.0

    def test_et0_higher_in_summer(self):
        """ET0 is higher in summer than winter at same location."""
        from src.digital_twin.water_balance import WaterBalanceModel

        wb = WaterBalanceModel()
        summer_weather = {
            "temp_max": 30.0,
            "temp_min": 20.0,
            "humidity_max": 80.0,
            "humidity_min": 40.0,
            "wind_speed": 2.0,
            "solar_radiation": 22.0,
            "latitude": 35.0,
            "day_of_year": 172,
        }
        winter_weather = {
            "temp_max": 15.0,
            "temp_min": 5.0,
            "humidity_max": 80.0,
            "humidity_min": 40.0,
            "wind_speed": 2.0,
            "solar_radiation": 10.0,
            "latitude": 35.0,
            "day_of_year": 355,
        }
        et0_summer = wb.compute_et0(summer_weather)
        et0_winter = wb.compute_et0(winter_weather)
        assert et0_summer > et0_winter

    def test_et0_higher_at_equator_in_winter(self):
        """ET0 is higher at equator than at high latitude in winter (shorter days at high lat)."""
        from src.digital_twin.water_balance import WaterBalanceModel

        wb = WaterBalanceModel()
        equator_weather = {
            "temp_max": 30.0,
            "temp_min": 20.0,
            "humidity_max": 80.0,
            "humidity_min": 40.0,
            "wind_speed": 2.0,
            "solar_radiation": 22.0,
            "latitude": 0.0,
            "day_of_year": 355,
        }
        high_lat_weather = {
            "temp_max": 30.0,
            "temp_min": 20.0,
            "humidity_max": 80.0,
            "humidity_min": 40.0,
            "wind_speed": 2.0,
            "solar_radiation": 22.0,
            "latitude": 60.0,
            "day_of_year": 355,
        }
        et0_equator = wb.compute_et0(equator_weather)
        et0_high = wb.compute_et0(high_lat_weather)
        assert et0_equator > et0_high
