"""Testes para módulos de analytics (SHI, PD Aggregation, Validation)."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.sector_monitor import SectorMonitor, WeightingMethod
from src.analytics.health_index import SectoralHealthIndex
from src.analytics.pd_aggregation import PDAggregation
from src.analytics.validation import ModelValidator


@pytest.fixture
def sample_dataset():
    """Dataset analítico de teste com 20 empresas em 3 setores."""
    np.random.seed(42)
    n = 20
    sectors = ['Energia', 'Bancos', 'Varejo']
    return pd.DataFrame({
        'cnpj_cia': [f'{i:014d}' for i in range(n)],
        'company_name': [f'Company_{i}' for i in range(n)],
        'sector': np.random.choice(sectors, n),
        'reference_date': '2024-12-31',
        'total_assets': np.random.uniform(1e6, 1e9, n),
        'equity': np.random.uniform(-1e6, 5e8, n),
        'current_assets': np.random.uniform(1e5, 3e8, n),
        'current_liabilities': np.random.uniform(1e5, 2e8, n),
        'net_revenue': np.random.uniform(1e5, 5e8, n),
        'ebit': np.random.uniform(-1e7, 1e8, n),
        'net_income': np.random.uniform(-5e7, 1e8, n),
        'short_term_debt': np.random.uniform(0, 1e8, n),
        'long_term_debt': np.random.uniform(0, 2e8, n),
        'roa': np.random.uniform(-0.2, 0.15, n),
        'roe': np.random.uniform(-0.5, 0.3, n),
        'current_ratio': np.random.uniform(0.3, 3.0, n),
        'debt_to_equity': np.random.uniform(-2, 10, n),
        'ebit_margin': np.random.uniform(-0.5, 0.3, n),
        'altman_zscore': np.random.uniform(-5, 10, n),
        'pd_merton': np.random.uniform(0.001, 0.5, n),
        'distance_to_default': np.random.uniform(-1, 8, n),
    })


@pytest.fixture
def sample_merton_results():
    """Resultados Merton de teste."""
    np.random.seed(42)
    n = 15
    return pd.DataFrame({
        'cnpj': [f'{i:014d}' for i in range(n)],
        'pd_merton': np.random.uniform(0.001, 0.3, n),
        'distance_to_default': np.random.uniform(0, 6, n),
        'converged': [True] * 13 + [False] * 2,
        'iterations': np.random.randint(3, 50, n),
    })


class TestSectorMonitor:

    def test_compute_sector_metrics(self, sample_dataset):
        monitor = SectorMonitor()
        result = monitor.compute_sector_metrics(sample_dataset)
        assert not result.empty
        assert 'sector' in result.columns
        assert len(result) == 3  # 3 setores

    def test_sector_metrics_columns(self, sample_dataset):
        monitor = SectorMonitor()
        result = monitor.compute_sector_metrics(sample_dataset)
        expected_cols = [
            'sector', 'n_companies', 'ebitda_medio_ponderado',
            'alavancagem_media', 'roa_medio',
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_weighting_methods(self, sample_dataset):
        eq = SectorMonitor(WeightingMethod.EQUAL)
        vw = SectorMonitor(WeightingMethod.VALUE)
        r1 = eq.compute_sector_metrics(sample_dataset)
        r2 = vw.compute_sector_metrics(sample_dataset)
        assert len(r1) == len(r2)

    def test_empty_dataframe(self):
        monitor = SectorMonitor()
        result = monitor.compute_sector_metrics(pd.DataFrame())
        assert result.empty

    def test_build_sector_index(self, sample_dataset):
        monitor = SectorMonitor()
        idx = monitor.build_sector_index(sample_dataset)
        # Com apenas uma data, o index base 100 não tem mudança
        if not idx.empty:
            assert 'index_value' in idx.columns


class TestSectoralHealthIndex:

    def test_compute_shi(self, sample_dataset):
        shi = SectoralHealthIndex()
        result = shi.compute(sample_dataset)
        assert not result.empty
        assert 'shi_normalized' in result.columns

    def test_shi_range(self, sample_dataset):
        shi = SectoralHealthIndex()
        result = shi.compute(sample_dataset)
        assert result['shi_normalized'].min() >= -3.0
        assert result['shi_normalized'].max() <= 3.0

    def test_shi_has_classification(self, sample_dataset):
        shi = SectoralHealthIndex()
        result = shi.compute(sample_dataset)
        assert 'classification' in result.columns
        valid_classes = {
            "Deterioração Severa", "Deterioração Moderada",
            "Deterioração Leve", "Neutro",
            "Melhoria Leve", "Melhoria Moderada",
            "Fortalecimento Relevante",
        }
        for cls in result['classification']:
            assert cls in valid_classes

    def test_shi_has_components(self, sample_dataset):
        shi = SectoralHealthIndex()
        result = shi.compute(sample_dataset)
        for col in ['z_rentabilidade', 'z_alavancagem', 'z_liquidez', 'z_volatilidade']:
            assert col in result.columns

    def test_shi_empty(self):
        shi = SectoralHealthIndex()
        result = shi.compute(pd.DataFrame())
        assert result.empty

    def test_shi_color(self, sample_dataset):
        shi = SectoralHealthIndex()
        result = shi.compute(sample_dataset)
        assert 'color' in result.columns
        for c in result['color']:
            assert c.startswith('#')


class TestPDAggregation:

    def test_aggregate_by_sector(self, sample_dataset):
        agg = PDAggregation()
        result = agg.aggregate_by_sector(sample_dataset)
        assert not result.empty
        assert 'pd_media' in result.columns
        assert 'pd_mediana' in result.columns
        assert 'pd_p90' in result.columns

    def test_rating_distribution(self, sample_dataset):
        agg = PDAggregation()
        result = agg.rating_distribution(sample_dataset)
        assert not result.empty
        assert 'rating' in result.columns
        assert 'pct' in result.columns
        assert abs(result['pct'].sum() - 100.0) < 0.01

    def test_sector_rating_matrix(self, sample_dataset):
        agg = PDAggregation()
        matrix = agg.sector_rating_matrix(sample_dataset)
        assert not matrix.empty
        assert matrix.shape[0] == 3  # 3 setores

    def test_concentration_metrics(self, sample_dataset):
        agg = PDAggregation()
        sector_agg = agg.aggregate_by_sector(sample_dataset)
        conc = agg.compute_concentration_metrics(sector_agg)
        assert 'hhi' in conc
        assert 'n_sectors' in conc
        assert conc['n_sectors'] == 3

    def test_investment_grade_pct(self, sample_dataset):
        agg = PDAggregation()
        result = agg.aggregate_by_sector(sample_dataset)
        for _, row in result.iterrows():
            assert 0 <= row['pct_investment_grade'] <= 1
            assert 0 <= row['pct_distress'] <= 1

    def test_empty_input(self):
        agg = PDAggregation()
        result = agg.aggregate_by_sector(pd.DataFrame())
        assert result.empty


class TestModelValidator:

    def test_backtest(self, sample_dataset):
        validator = ModelValidator()
        result = validator.backtest(sample_dataset)
        assert result.n_total > 0
        assert 0 <= result.ks_statistic <= 1
        assert -1 <= result.gini_coefficient <= 1

    def test_sensitivity_analysis(self, sample_dataset):
        validator = ModelValidator()
        result = validator.sensitivity_analysis(sample_dataset)
        assert not result.empty
        assert 'cenario' in result.columns
        assert len(result) >= 5  # default scenarios

    def test_sensitivity_custom_shocks(self, sample_dataset):
        validator = ModelValidator()
        shocks = {'Base': 1.0, 'Stress': 2.0}
        result = validator.sensitivity_analysis(sample_dataset, shocks=shocks)
        assert len(result) == 2

    def test_correlation_with_defaults(self, sample_dataset):
        validator = ModelValidator()
        result = validator.correlation_with_defaults(sample_dataset)
        assert 'rank_correlation' in result
        assert 'n_rj' in result

    def test_decile_analysis(self, sample_dataset):
        validator = ModelValidator()
        bt = validator.backtest(sample_dataset)
        if not bt.decile_analysis.empty:
            assert 'decile' in bt.decile_analysis.columns
            assert 'n_defaults' in bt.decile_analysis.columns

    def test_backtest_empty(self):
        validator = ModelValidator()
        result = validator.backtest(pd.DataFrame())
        assert result.n_total == 0
