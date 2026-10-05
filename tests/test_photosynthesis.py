"""Tests for photosynthesis model: C3/C4 pathways, radiation use efficiency."""

import pytest

from src.digital_twin.photosynthesis import (
    C3C4Pathway,
    PhotosynthesisModel,
    PhotosynthesisState,
)


class TestRUEBiomassAccumulation:
    """Radiation use efficiency biomass accumulation."""

    def test_rue_biomass_positive_under_full_light(self):
        """Biomass accumulates under full light."""
        model = PhotosynthesisModel()
        result = model.rue_biomass(
            par_mj_m2_day=20.0,
            rue=2.5,
            light_interception=1.0,
        )
        assert result.biomass_g_m2 > 0.0
        assert result.apar_mj_m2 == pytest.approx(20.0)

    def test_rue_biomass_scales_with_par(self):
        """Doubling PAR doubles biomass (linear regime)."""
        model = PhotosynthesisModel()
        r1 = model.rue_biomass(par_mj_m2_day=10.0, rue=2.5, light_interception=1.0)
        r2 = model.rue_biomass(par_mj_m2_day=20.0, rue=2.5, light_interception=1.0)
        assert r2.biomass_g_m2 == pytest.approx(2.0 * r1.biomass_g_m2)

    def test_rue_biomass_scales_with_interception(self):
        """Biomass scales with light interception fraction."""
        model = PhotosynthesisModel()
        r1 = model.rue_biomass(par_mj_m2_day=20.0, rue=2.5, light_interception=0.5)
        r2 = model.rue_biomass(par_mj_m2_day=20.0, rue=2.5, light_interception=1.0)
        assert r2.biomass_g_m2 == pytest.approx(2.0 * r1.biomass_g_m2)

    def test_rue_biomass_zero_at_zero_interception(self):
        """No biomass when no light is intercepted."""
        model = PhotosynthesisModel()
        result = model.rue_biomass(par_mj_m2_day=20.0, rue=2.5, light_interception=0.0)
        assert result.biomass_g_m2 == 0.0

    def test_rue_biomass_zero_at_zero_par(self):
        """No biomass when no PAR."""
        model = PhotosynthesisModel()
        result = model.rue_biomass(par_mj_m2_day=0.0, rue=2.5, light_interception=1.0)
        assert result.biomass_g_m2 == 0.0

    def test_rue_biomass_scales_with_rue(self):
        """Higher RUE produces more biomass."""
        model = PhotosynthesisModel()
        r1 = model.rue_biomass(par_mj_m2_day=20.0, rue=1.5, light_interception=1.0)
        r2 = model.rue_biomass(par_mj_m2_day=20.0, rue=3.0, light_interception=1.0)
        assert r2.biomass_g_m2 == pytest.approx(2.0 * r1.biomass_g_m2)


class TestC3C4PathwayDifferences:
    """C3 vs C4 pathway differences."""

    def test_c4_higher_rue_than_c3(self):
        """C4 plants have higher radiation use efficiency."""
        model = PhotosynthesisModel()
        c3 = model.rue_biomass(
            par_mj_m2_day=20.0,
            rue=C3C4Pathway.C3.default_rue(),
            light_interception=1.0,
        )
        c4 = model.rue_biomass(
            par_mj_m2_day=20.0,
            rue=C3C4Pathway.C4.default_rue(),
            light_interception=1.0,
        )
        assert c4.biomass_g_m2 > c3.biomass_g_m2

    def test_c4_lower_quantum_efficiency(self):
        """C4 plants have lower quantum efficiency (extra ATP cost for CO2 concentration)."""
        model = PhotosynthesisModel()
        qe_c3 = model.quantum_efficiency(C3C4Pathway.C3)
        qe_c4 = model.quantum_efficiency(C3C4Pathway.C4)
        assert qe_c4 < qe_c3

    def test_c4_higher_co2_compensation_point(self):
        """C4 plants have lower CO2 compensation point (CO2 concentrating mechanism)."""
        model = PhotosynthesisModel()
        gamma_c3 = model.co2_compensation_point(C3C4Pathway.C3)
        gamma_c4 = model.co2_compensation_point(C3C4Pathway.C4)
        assert gamma_c4 < gamma_c3

    def test_c4_less_co2_responsive(self):
        """C4 plants show less response to elevated CO2 (saturated Rubisco)."""
        model = PhotosynthesisModel()
        resp_c3 = model.co2_response_factor(co2_ppm=600.0, pathway=C3C4Pathway.C3, base_co2=400.0)
        resp_c4 = model.co2_response_factor(co2_ppm=600.0, pathway=C3C4Pathway.C4, base_co2=400.0)
        assert resp_c3 > resp_c4

    def test_c4_higher_optimal_temperature(self):
        """C4 plants have higher optimal temperature."""
        model = PhotosynthesisModel()
        t_opt_c3 = model.optimal_temperature(C3C4Pathway.C3)
        t_opt_c4 = model.optimal_temperature(C3C4Pathway.C4)
        assert t_opt_c4 > t_opt_c3


class TestTemperatureResponse:
    """Temperature response of photosynthesis."""

    def test_optimal_temperature_gives_max_rate(self):
        """Photosynthesis peaks at optimal temperature."""
        model = PhotosynthesisModel()
        t_opt = model.optimal_temperature(C3C4Pathway.C3)
        f_opt = model.temperature_response(t_opt, C3C4Pathway.C3)
        assert f_opt == pytest.approx(1.0, abs=0.01)

    def test_zero_at_extreme_temperatures(self):
        """Photosynthesis stops at extreme temperatures."""
        model = PhotosynthesisModel()
        assert model.temperature_response(-10.0, C3C4Pathway.C3) == 0.0
        assert model.temperature_response(50.0, C3C4Pathway.C3) == 0.0

    def test_reduced_at_suboptimal_temperature(self):
        """Photosynthesis reduced below optimum."""
        model = PhotosynthesisModel()
        t_opt = model.optimal_temperature(C3C4Pathway.C3)
        f_cold = model.temperature_response(t_opt - 10.0, C3C4Pathway.C3)
        f_hot = model.temperature_response(t_opt + 10.0, C3C4Pathway.C3)
        assert f_cold < 1.0
        assert f_hot < 1.0

    def test_c4_higher_temperature_range(self):
        """C4 plants have higher temperature range (tropical origin)."""
        model = PhotosynthesisModel()
        t_min_c3 = model.min_temperature(C3C4Pathway.C3)
        t_min_c4 = model.min_temperature(C3C4Pathway.C4)
        t_max_c3 = model.max_temperature(C3C4Pathway.C3)
        t_max_c4 = model.max_temperature(C3C4Pathway.C4)
        assert t_min_c4 > t_min_c3
        assert t_max_c4 > t_max_c3


class TestLightResponse:
    """Light response curve (rectangular hyperbola)."""

    def test_saturates_at_high_light(self):
        """Photosynthesis saturates at high light."""
        model = PhotosynthesisModel()
        a_low = model.light_response(par_umol_m2_s=200.0, pmax=25.0)
        a_mid = model.light_response(par_umol_m2_s=1000.0, pmax=25.0)
        a_high = model.light_response(par_umol_m2_s=2000.0, pmax=25.0)
        assert a_low < a_mid < a_high
        assert a_high < 25.0 * 1.05  # approaches but doesn't exceed Pmax significantly

    def test_zero_at_zero_light(self):
        """No photosynthesis in darkness."""
        model = PhotosynthesisModel()
        assert model.light_response(par_umol_m2_s=0.0, pmax=25.0) == 0.0

    def test_increases_with_pmax(self):
        """Higher Pmax gives higher photosynthesis at same light."""
        model = PhotosynthesisModel()
        a1 = model.light_response(par_umol_m2_s=1000.0, pmax=15.0)
        a2 = model.light_response(par_umol_m2_s=1000.0, pmax=30.0)
        assert a2 > a1


class TestFarquharModel:
    """Farquhar-von Caemmerer-Berry biochemical model."""

    def test_light_limited_at_low_light(self):
        """At low light, photosynthesis is light-limited (Aj < Ac)."""
        model = PhotosynthesisModel()
        result = model.farquhar_gross_photosynthesis(
            par_umol_m2_s=100.0,
            ci_umol_mol=300.0,
            vcmax=80.0,
            jmax=160.0,
            temp_c=25.0,
        )
        assert result.limiting_process == "light"

    def test_rubisco_limited_at_high_light(self):
        """At high light, photosynthesis is Rubisco-limited (Ac < Aj)."""
        model = PhotosynthesisModel()
        result = model.farquhar_gross_photosynthesis(
            par_umol_m2_s=2000.0,
            ci_umol_mol=300.0,
            vcmax=40.0,
            jmax=160.0,
            temp_c=25.0,
        )
        assert result.limiting_process == "rubisco"

    def test_increases_with_co2(self):
        """Gross photosynthesis increases with CO2 concentration."""
        model = PhotosynthesisModel()
        a1 = model.farquhar_gross_photosynthesis(
            par_umol_m2_s=1500.0,
            ci_umol_mol=200.0,
            vcmax=80.0,
            jmax=160.0,
            temp_c=25.0,
        )
        a2 = model.farquhar_gross_photosynthesis(
            par_umol_m2_s=1500.0,
            ci_umol_mol=400.0,
            vcmax=80.0,
            jmax=160.0,
            temp_c=25.0,
        )
        assert a2.gross_photosynthesis > a1.gross_photosynthesis

    def test_increases_with_light(self):
        """Gross photosynthesis increases with light."""
        model = PhotosynthesisModel()
        a1 = model.farquhar_gross_photosynthesis(
            par_umol_m2_s=200.0,
            ci_umol_mol=300.0,
            vcmax=80.0,
            jmax=160.0,
            temp_c=25.0,
        )
        a2 = model.farquhar_gross_photosynthesis(
            par_umol_m2_s=1500.0,
            ci_umol_mol=300.0,
            vcmax=80.0,
            jmax=160.0,
            temp_c=25.0,
        )
        assert a2.gross_photosynthesis > a1.gross_photosynthesis

    def test_net_photosynthesis_subtracts_respiration(self):
        """Net photosynthesis = gross - dark respiration."""
        model = PhotosynthesisModel()
        result = model.farquhar_gross_photosynthesis(
            par_umol_m2_s=1500.0,
            ci_umol_mol=300.0,
            vcmax=80.0,
            jmax=160.0,
            temp_c=25.0,
        )
        assert result.net_photosynthesis < result.gross_photosynthesis
        assert result.net_photosynthesis > 0.0

    def test_co2_compensation_point(self):
        """At compensation point, net photosynthesis is zero."""
        model = PhotosynthesisModel()
        gamma = model.co2_compensation_point(C3C4Pathway.C3)
        result = model.farquhar_gross_photosynthesis(
            par_umol_m2_s=1500.0,
            ci_umol_mol=gamma,
            vcmax=80.0,
            jmax=160.0,
            temp_c=25.0,
        )
        assert result.net_photosynthesis == pytest.approx(0.0, abs=0.5)


class TestWaterStressEffect:
    """Water stress reduces photosynthesis."""

    def test_water_stress_reduces_rue(self):
        """Water stress reduces effective RUE."""
        model = PhotosynthesisModel()
        r_full = model.rue_biomass(par_mj_m2_day=20.0, rue=2.5, light_interception=1.0)
        r_stress = model.rue_biomass(
            par_mj_m2_day=20.0, rue=2.5, light_interception=1.0, water_stress_factor=0.5
        )
        assert r_stress.biomass_g_m2 < r_full.biomass_g_m2

    def test_severe_water_stress_near_zero(self):
        """Severe water stress nearly stops photosynthesis."""
        model = PhotosynthesisModel()
        result = model.rue_biomass(
            par_mj_m2_day=20.0, rue=2.5, light_interception=1.0, water_stress_factor=0.01
        )
        assert result.biomass_g_m2 < 1.0

    def test_no_water_stress_no_reduction(self):
        """No water stress means no reduction."""
        model = PhotosynthesisModel()
        r1 = model.rue_biomass(par_mj_m2_day=20.0, rue=2.5, light_interception=1.0)
        r2 = model.rue_biomass(
            par_mj_m2_day=20.0, rue=2.5, light_interception=1.0, water_stress_factor=1.0
        )
        assert r1.biomass_g_m2 == pytest.approx(r2.biomass_g_m2)


class TestDailySimulation:
    """Daily photosynthesis simulation."""

    def test_biomass_accumulates_over_days(self):
        """Biomass accumulates over multiple days."""
        model = PhotosynthesisModel()
        state = PhotosynthesisState(lai=3.0, biomass_g_m2=100.0)
        result = model.simulate(
            initial_state=state,
            days=10,
            daily_par_mj_m2=20.0,
            pathway=C3C4Pathway.C3,
        )
        assert result.days_simulated == 10
        assert result.final_state.biomass_g_m2 > 100.0
        assert len(result.history) == 10

    def test_c4_produces_more_biomass_at_high_temp(self):
        """C4 produces more biomass than C3 at high temperature (35°C)."""
        model = PhotosynthesisModel()
        state_c3 = PhotosynthesisState(lai=3.0, biomass_g_m2=100.0, temperature=35.0)
        state_c4 = PhotosynthesisState(lai=3.0, biomass_g_m2=100.0, temperature=35.0)
        r_c3 = model.simulate(
            initial_state=state_c3, days=10, daily_par_mj_m2=20.0, pathway=C3C4Pathway.C3
        )
        r_c4 = model.simulate(
            initial_state=state_c4, days=10, daily_par_mj_m2=20.0, pathway=C3C4Pathway.C4
        )
        assert r_c4.final_state.biomass_g_m2 > r_c3.final_state.biomass_g_m2

    def test_lai_increases_light_interception(self):
        """Higher LAI increases light interception."""
        model = PhotosynthesisModel()
        f_low = model.light_interception(lai=1.0)
        f_high = model.light_interception(lai=4.0)
        assert f_high > f_low
        assert f_high <= 1.0

    def test_simulation_with_water_stress(self):
        """Water stress reduces total biomass accumulation."""
        model = PhotosynthesisModel()
        state = PhotosynthesisState(lai=3.0, biomass_g_m2=100.0)
        r_normal = model.simulate(
            initial_state=state, days=10, daily_par_mj_m2=20.0, pathway=C3C4Pathway.C3
        )
        r_stress = model.simulate(
            initial_state=state,
            days=10,
            daily_par_mj_m2=20.0,
            pathway=C3C4Pathway.C3,
            water_stress_factor=0.4,
        )
        assert r_stress.final_state.biomass_g_m2 < r_normal.final_state.biomass_g_m2

    def test_zero_days_returns_initial_state(self):
        """Zero simulation days returns initial state."""
        model = PhotosynthesisModel()
        state = PhotosynthesisState(lai=3.0, biomass_g_m2=100.0)
        result = model.simulate(
            initial_state=state, days=0, daily_par_mj_m2=20.0, pathway=C3C4Pathway.C3
        )
        assert result.days_simulated == 0
        assert result.final_state.biomass_g_m2 == 100.0
        assert len(result.history) == 0

    def test_negative_days_raises(self):
        """Negative days raises ValueError."""
        model = PhotosynthesisModel()
        state = PhotosynthesisState(lai=3.0, biomass_g_m2=100.0)
        with pytest.raises(ValueError, match="Days must be non-negative"):
            model.simulate(
                initial_state=state, days=-1, daily_par_mj_m2=20.0, pathway=C3C4Pathway.C3
            )
