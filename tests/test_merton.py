"""Testes para o modelo estrutural de Merton."""

import numpy as np
import pandas as pd
import pytest

from src.models.merton_model import (
    MertonModel, MertonInput, MertonResult, DriftMode,
)


@pytest.fixture
def model():
    return MertonModel(max_iterations=200, tolerance=1e-8)


@pytest.fixture
def sample_input():
    """Empresa hipotética: E=100M, σE=40%, D(CP)=30M, D(LP)=50M."""
    return MertonInput(
        cnpj="11.111.111/0001-11",
        company_name="Empresa Teste S.A.",
        equity_value=100_000_000,
        equity_volatility=0.40,
        short_term_debt=30_000_000,
        long_term_debt=50_000_000,
        risk_free_rate=0.1375,
        time_horizon=1.0,
        drift_mode=DriftMode.RISK_NEUTRAL,
        sector="Teste",
        reference_date="2024-12-31",
    )


class TestMertonModel:

    def test_basic_convergence(self, model, sample_input):
        result = model.compute_pd(sample_input)
        assert result.converged is True
        assert result.iterations < 200

    def test_pd_range(self, model, sample_input):
        result = model.compute_pd(sample_input)
        assert 0 < result.pd_merton < 1

    def test_dd_positive_for_healthy(self, model, sample_input):
        """Empresa saudável (E >> D) deve ter DD > 0."""
        result = model.compute_pd(sample_input)
        assert result.distance_to_default > 0

    def test_asset_value_greater_than_equity(self, model, sample_input):
        """V deve ser > E (ativos = equity + dívida)."""
        result = model.compute_pd(sample_input)
        assert result.asset_value > sample_input.equity_value

    def test_asset_volatility_less_than_equity_volatility(self, model, sample_input):
        """σV < σE (efeito de alavancagem)."""
        result = model.compute_pd(sample_input)
        assert result.asset_volatility < sample_input.equity_volatility

    def test_high_debt_higher_pd(self, model):
        """Mais dívida → maior PD."""
        low_debt = MertonInput(
            cnpj="1", company_name="Low Debt",
            equity_value=100, equity_volatility=0.3,
            short_term_debt=10, long_term_debt=20,
            risk_free_rate=0.10, time_horizon=1.0,
        )
        high_debt = MertonInput(
            cnpj="2", company_name="High Debt",
            equity_value=100, equity_volatility=0.3,
            short_term_debt=80, long_term_debt=120,
            risk_free_rate=0.10, time_horizon=1.0,
        )
        r1 = model.compute_pd(low_debt)
        r2 = model.compute_pd(high_debt)
        assert r2.pd_merton > r1.pd_merton

    def test_higher_volatility_higher_pd(self, model):
        """Maior volatilidade → maior PD."""
        low_vol = MertonInput(
            cnpj="1", company_name="Low Vol",
            equity_value=100, equity_volatility=0.15,
            short_term_debt=30, long_term_debt=40,
            risk_free_rate=0.10, time_horizon=1.0,
        )
        high_vol = MertonInput(
            cnpj="2", company_name="High Vol",
            equity_value=100, equity_volatility=0.80,
            short_term_debt=30, long_term_debt=40,
            risk_free_rate=0.10, time_horizon=1.0,
        )
        r1 = model.compute_pd(low_vol)
        r2 = model.compute_pd(high_vol)
        assert r2.pd_merton > r1.pd_merton

    def test_debt_barrier_calculation(self, model, sample_input):
        """D = CP + 0.5 * LP."""
        result = model.compute_pd(sample_input)
        expected = 30_000_000 + 0.5 * 50_000_000
        assert result.debt_barrier == expected

    def test_rating_assignment(self, model, sample_input):
        result = model.compute_pd(sample_input)
        assert result.rating_bucket in [
            "AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "D"
        ]

    def test_invalid_equity_zero(self, model):
        inp = MertonInput(
            cnpj="1", company_name="Zero Equity",
            equity_value=0, equity_volatility=0.3,
            short_term_debt=10, long_term_debt=20,
            risk_free_rate=0.10, time_horizon=1.0,
        )
        result = model.compute_pd(inp)
        assert not result.converged or np.isnan(result.pd_merton)

    def test_invalid_negative_debt(self, model):
        inp = MertonInput(
            cnpj="1", company_name="No Debt",
            equity_value=100, equity_volatility=0.3,
            short_term_debt=-5, long_term_debt=0,
            risk_free_rate=0.10, time_horizon=1.0,
        )
        result = model.compute_pd(inp)
        assert np.isnan(result.pd_merton) or not result.converged

    def test_drift_modes(self, model):
        base = MertonInput(
            cnpj="1", company_name="Test",
            equity_value=100, equity_volatility=0.3,
            short_term_debt=30, long_term_debt=40,
            risk_free_rate=0.10, time_horizon=1.0,
        )

        for mode in DriftMode:
            base_copy = MertonInput(
                cnpj=base.cnpj, company_name=base.company_name,
                equity_value=base.equity_value,
                equity_volatility=base.equity_volatility,
                short_term_debt=base.short_term_debt,
                long_term_debt=base.long_term_debt,
                risk_free_rate=base.risk_free_rate,
                time_horizon=base.time_horizon,
                drift_mode=mode,
                historical_return=0.08,
            )
            result = model.compute_pd(base_copy)
            assert result.converged
            assert 0 < result.pd_merton < 1

    def test_batch_computation(self, model):
        inputs = [
            MertonInput(
                cnpj=f"{i}", company_name=f"Company {i}",
                equity_value=100 + i * 10,
                equity_volatility=0.3,
                short_term_debt=20, long_term_debt=30,
                risk_free_rate=0.10, time_horizon=1.0,
            )
            for i in range(10)
        ]
        result_df = model.compute_batch(inputs)
        assert len(result_df) == 10
        assert 'pd_merton' in result_df.columns
        assert 'rating_bucket' in result_df.columns

    def test_estimate_equity_volatility(self):
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 252))
        vol = MertonModel.estimate_equity_volatility(returns)
        assert 0.2 < vol < 0.5  # ~31% anualizado

    def test_estimate_volatility_insufficient_data(self):
        returns = pd.Series([0.01, 0.02])
        vol = MertonModel.estimate_equity_volatility(returns, min_periods=60)
        assert np.isnan(vol)

    def test_prepare_inputs_from_fundamentals(self):
        df = pd.DataFrame({
            'cnpj_cia': ['111', '222'],
            'company_name': ['A', 'B'],
            'equity': [100, 200],
            'short_term_debt': [30, 50],
            'long_term_debt': [40, 60],
            'total_assets': [200, 400],
            'roa': [0.05, 0.08],
            'sector': ['Teste', 'Teste'],
            'reference_date': ['2024-12-31', '2024-12-31'],
        })
        inputs = MertonModel.prepare_inputs_from_fundamentals(df)
        assert len(inputs) == 2
        assert inputs[0].equity_value == 100
        assert inputs[0].short_term_debt == 30

    def test_leverage_ratio(self, model, sample_input):
        result = model.compute_pd(sample_input)
        assert 0 < result.leverage_ratio < 1  # D/V < 1 para empresa saudável

    def test_d1_d2_relationship(self, model, sample_input):
        """d2 = d1 - σV√T."""
        result = model.compute_pd(sample_input)
        expected_d2 = result.d1 - result.asset_volatility * np.sqrt(1.0)
        assert abs(result.d2 - expected_d2) < 1e-6
