"""
Testes do Motor de Data Quality.
Cobre: integridade de balanço, completude, EBITDA, preços, OHLC.
"""

import numpy as np
import pandas as pd
import pytest

from src.quality.data_quality import DataQualityEngine, QualityReport


@pytest.fixture
def dq_engine():
    return DataQualityEngine()


@pytest.fixture
def balanced_financial_df():
    """DataFrame financeiro com balanço equilibrado."""
    return pd.DataFrame([
        {
            "cnpj_cia": "33000167000101",
            "reference_date": "2023-12-31",
            "company_name": "PETROBRAS SA",
            "total_assets": 1_000_000,
            "total_liabilities_equity": 1_000_000,
            "equity": 300_000,
            "current_assets": 400_000,
            "current_liabilities": 200_000,
            "net_revenue": 500_000,
            "net_income": 50_000,
            "ebit": 80_000,
            "ebitda_approx": 120_000,
            "altman_zscore": 3.5,
            "bs_check_pct": 0.0,
            "bs_balanced": True,
            "completeness_score": 0.95,
            "missing_pct": 0.05,
        }
    ])


@pytest.fixture
def unbalanced_financial_df():
    """DataFrame com balanço desbalanceado (desvio de 5%)."""
    return pd.DataFrame([
        {
            "cnpj_cia": "11111111000100",
            "reference_date": "2023-12-31",
            "company_name": "EMPRESA RUIM SA",
            "total_assets": 1_000_000,
            "total_liabilities_equity": 950_000,  # Diferença de 5%
            "equity": 100_000,
            "net_revenue": 200_000,
            "net_income": -50_000,
            "ebit": -30_000,
            "ebitda_approx": -10_000,
            "altman_zscore": 0.5,
            "bs_check_pct": 0.05,   # 5% > tolerância de 1%
            "bs_balanced": False,
            "completeness_score": 0.60,  # Abaixo do mínimo
            "missing_pct": 0.40,
        }
    ])


@pytest.fixture
def market_df():
    """DataFrame de cotações para testes."""
    return pd.DataFrame([
        {"ticker": "PETR4", "date": "2024-01-02", "open": 36.5, "high": 37.2,
         "low": 35.8, "close": 36.9, "volume_financial": 1_000_000},
        {"ticker": "PETR4", "date": "2024-01-03", "open": 36.9, "high": 37.5,
         "low": 36.0, "close": 37.2, "volume_financial": 800_000},
        {"ticker": "VALE3", "date": "2024-01-02", "open": 70.0, "high": 71.0,
         "low": 69.0, "close": 70.5, "volume_financial": 2_000_000},
    ])


# =============================================================================
# Testes Financeiros
# =============================================================================

class TestFinancialChecks:

    def test_balanced_bs_passes(self, dq_engine, balanced_financial_df):
        """Balanço equilibrado não deve gerar erros."""
        report = dq_engine.run_financial_checks(balanced_financial_df)
        bs_errors = [i for i in report.errors if i.check_name == "balance_sheet_integrity"]
        assert len(bs_errors) == 0

    def test_unbalanced_bs_generates_error(self, dq_engine, unbalanced_financial_df):
        """Balanço desbalanceado (>1%) deve gerar erro."""
        report = dq_engine.run_financial_checks(unbalanced_financial_df)
        bs_errors = [i for i in report.errors if i.check_name == "balance_sheet_integrity"]
        assert len(bs_errors) >= 1
        assert bs_errors[0].value > DataQualityEngine.BS_TOLERANCE

    def test_low_completeness_generates_warning(self, dq_engine, unbalanced_financial_df):
        """Completude abaixo do mínimo deve gerar warning."""
        report = dq_engine.run_financial_checks(unbalanced_financial_df)
        completeness_warnings = [
            i for i in report.warnings if i.check_name == "data_completeness"
        ]
        assert len(completeness_warnings) >= 1

    def test_high_completeness_no_warning(self, dq_engine, balanced_financial_df):
        """Completude alta não deve gerar warning."""
        report = dq_engine.run_financial_checks(balanced_financial_df)
        completeness_warnings = [
            i for i in report.warnings if i.check_name == "data_completeness"
        ]
        assert len(completeness_warnings) == 0

    def test_ebitda_inconsistency_detected(self, dq_engine):
        """EBITDA < EBIT deve gerar warning."""
        df = pd.DataFrame([{
            "cnpj_cia": "12345678000100",
            "reference_date": "2023-12-31",
            "ebit": 100_000,
            "ebitda_approx": 90_000,  # EBITDA < EBIT → inconsistência
            "completeness_score": 0.9,
            "bs_check_pct": 0.0,
            "bs_balanced": True,
        }])
        report = dq_engine.run_financial_checks(df)
        ebitda_warnings = [i for i in report.warnings if i.check_name == "ebitda_consistency"]
        assert len(ebitda_warnings) >= 1

    def test_negative_equity_flagged_as_info(self, dq_engine):
        """PL negativo deve ser flagado como info (distress, não erro)."""
        df = pd.DataFrame([{
            "cnpj_cia": "12345678000100",
            "reference_date": "2023-12-31",
            "equity": -50_000,
            "completeness_score": 0.9,
            "bs_check_pct": 0.0,
        }])
        report = dq_engine.run_financial_checks(df)
        neg_eq_info = [i for i in report.issues if i.check_name == "negative_equity"]
        assert len(neg_eq_info) >= 1
        assert neg_eq_info[0].severity == "info"

    def test_future_reference_date_error(self, dq_engine):
        """Data de referência futura deve gerar erro (look-ahead bias)."""
        df = pd.DataFrame([{
            "cnpj_cia": "12345678000100",
            "reference_date": "2030-12-31",  # data futura
            "completeness_score": 0.9,
            "bs_check_pct": 0.0,
        }])
        report = dq_engine.run_financial_checks(df)
        future_errors = [i for i in report.errors if i.check_name == "future_reference_date"]
        assert len(future_errors) >= 1

    def test_overall_score_good_data(self, dq_engine, balanced_financial_df):
        """Score geral deve ser alto para dados bons."""
        report = dq_engine.run_financial_checks(balanced_financial_df)
        assert report.overall_score > 0.7

    def test_report_passed_with_only_warnings(self, dq_engine, unbalanced_financial_df):
        """Relatório sem erros críticos deve passar (mesmo com warnings)."""
        # Ajusta para ter só warnings
        df = unbalanced_financial_df.copy()
        df["bs_check_pct"] = 0.0  # Remove o erro de balanço
        df["bs_balanced"] = True
        report = dq_engine.run_financial_checks(df)
        # passed=True se não há erros
        bs_errors = [i for i in report.errors if i.check_name == "balance_sheet_integrity"]
        if not bs_errors:
            assert report.passed is True


# =============================================================================
# Testes de Mercado
# =============================================================================

class TestMarketChecks:

    def test_clean_market_data_no_errors(self, dq_engine, market_df):
        """Dados de mercado limpos não devem gerar erros."""
        report = dq_engine.run_market_checks(market_df)
        price_errors = [i for i in report.errors if i.check_name == "negative_price"]
        assert len(price_errors) == 0

    def test_negative_price_error(self, dq_engine):
        """Preço negativo deve gerar erro."""
        df = pd.DataFrame([{
            "ticker": "XXXX3",
            "date": "2024-01-02",
            "open": -1.0,
            "high": 5.0,
            "low": -1.0,
            "close": 3.0,
            "volume_financial": 100_000,
        }])
        report = dq_engine.run_market_checks(df)
        neg_errors = [i for i in report.errors if i.check_name == "negative_price"]
        assert len(neg_errors) >= 1

    def test_ohlc_consistency_check(self, dq_engine):
        """High < Low deve ser detectado como erro de OHLC."""
        df = pd.DataFrame([{
            "ticker": "XXXX3",
            "date": "2024-01-02",
            "open": 10.0,
            "high": 8.0,   # high < low → inválido
            "low": 9.0,
            "close": 9.5,
            "volume_financial": 100_000,
        }])
        report = dq_engine.run_market_checks(df)
        ohlc_errors = [i for i in report.errors if i.check_name == "ohlc_consistency"]
        assert len(ohlc_errors) >= 1

    def test_ohlc_valid_passes(self, dq_engine, market_df):
        """OHLC válido não deve gerar erro."""
        report = dq_engine.run_market_checks(market_df)
        ohlc_errors = [i for i in report.errors if i.check_name == "ohlc_consistency"]
        assert len(ohlc_errors) == 0

    def test_price_spike_detection(self, dq_engine):
        """Variação diária > 50% deve gerar warning."""
        df = pd.DataFrame([
            {"ticker": "XXXX3", "date": "2024-01-02", "close": 10.0, "volume_financial": 100_000,
             "open": 10.0, "high": 10.0, "low": 10.0},
            {"ticker": "XXXX3", "date": "2024-01-03", "close": 18.0, "volume_financial": 100_000,
             "open": 10.0, "high": 18.0, "low": 10.0},  # +80% em 1 dia
        ])
        report = dq_engine.run_market_checks(df)
        spike_warnings = [i for i in report.warnings if i.check_name == "price_spike"]
        assert len(spike_warnings) >= 1


# =============================================================================
# Testes de QualityReport
# =============================================================================

class TestQualityReport:

    def test_report_to_dataframe(self, dq_engine, unbalanced_financial_df):
        """Relatório deve ser convertível em DataFrame."""
        report = dq_engine.run_financial_checks(unbalanced_financial_df)
        df = report.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "check_name" in df.columns
        assert "severity" in df.columns

    def test_report_summary_structure(self, dq_engine, balanced_financial_df):
        """Summary do relatório deve ter campos obrigatórios."""
        report = dq_engine.run_financial_checks(balanced_financial_df)
        summary = report.summary()
        required = ["run_id", "dataset", "total_records", "errors", "warnings",
                    "overall_score", "passed"]
        for field in required:
            assert field in summary, f"Campo '{field}' ausente no summary"

    def test_company_scores_computed(self, dq_engine, balanced_financial_df):
        """Scores por empresa devem ser calculados."""
        report = dq_engine.run_financial_checks(balanced_financial_df)
        assert len(report.company_scores) > 0
        for cnpj, score in report.company_scores.items():
            assert "completeness" in score
            assert "consistency" in score
            assert 0 <= score["overall"] <= 1
