"""Tests for FAO-56 evapotranspiration calculation."""

from src.digital_twin.water_balance import WaterBalanceModel, WaterBalanceState


class TestFAO56ET:
    """Test FAO-56 Penman-Monteith evapotranspiration."""

    def test_compute_et0_basic(self):
        """Compute reference ET from weather data."""
        wb = WaterBalanceModel()
        weather = {
            "temp_max": 30.0,
            "temp_min": 20.0,
            "humidity_max": 80.0,
            "humidity_min": 40.0,
            "wind_speed": 2.0,
            "solar_radiation": 22.0,  # MJ/m²/day
            "latitude": 35.0,
            "day_of_year": 180,
        }
        et0 = wb.compute_et0(weather)
        assert et0 > 0.0
        assert et0 < 15.0  # Reasonable range for daily ET0

    def test_compute_et0_hot_dry(self):
        """Hot dry conditions produce higher ET0."""
        wb = WaterBalanceModel()
        hot_dry = {
            "temp_max": 38.0,
            "temp_min": 25.0,
            "humidity_max": 60.0,
            "humidity_min": 20.0,
            "wind_speed": 3.0,
            "solar_radiation": 25.0,
            "latitude": 35.0,
            "day_of_year": 180,
        }
        cool_wet = {
            "temp_max": 20.0,
            "temp_min": 15.0,
            "humidity_max": 90.0,
            "humidity_min": 70.0,
            "wind_speed": 1.0,
            "solar_radiation": 15.0,
            "latitude": 35.0,
            "day_of_year": 180,
        }
        et0_hot = wb.compute_et0(hot_dry)
        et0_cool = wb.compute_et0(cool_wet)
        assert et0_hot > et0_cool

    def test_crop_coefficient(self):
        """Crop coefficient adjusts ET0 to crop ET."""
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
        kc = 1.2
        etc = wb.crop_evapotranspiration(et0, kc)
        assert etc == et0 * kc

    def test_dual_crop_coefficient(self):
        """Dual Kcb + Ke method."""
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
        kcb = 1.1
        ke = 0.1
        etc = wb.dual_crop_coefficient_et(et0, kcb, ke)
        assert etc == et0 * (kcb + ke)

    def test_kc_by_growth_stage(self):
        """Kc varies by growth stage."""
        wb = WaterBalanceModel()
        kc_initial = wb.get_kc_for_stage("initial")
        kc_mid = wb.get_kc_for_stage("mid_season")
        kc_late = wb.get_kc_for_stage("late_season")
        # Mid-season Kc should be highest
        assert kc_mid > kc_initial
        assert kc_mid > kc_late

    def test_et_with_weather_data_integration(self):
        """Water balance can use computed ET from weather."""
        wb = WaterBalanceModel()
        state = WaterBalanceState(soil_moisture=0.3)
        weather = [
            {
                "temp_max": 30.0,
                "temp_min": 20.0,
                "humidity_max": 80.0,
                "humidity_min": 40.0,
                "wind_speed": 2.0,
                "solar_radiation": 22.0,
                "latitude": 35.0,
                "day_of_year": 180,
                "precipitation": 0.0,
            }
        ]
        # Use computed ET instead of provided ET
        et0 = wb.compute_et0(weather[0])
        etc = wb.crop_evapotranspiration(et0, kc=1.0)
        weather[0]["evapotranspiration"] = etc
        result = wb.simulate(state, weather)
        assert result.total_evapotranspiration > 0
