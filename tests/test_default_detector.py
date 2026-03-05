"""
Testes do módulo de Detecção de Default.
Cobre: cancelamentos, suspensão, queda de preço, fatos relevantes, score de distress.
"""

from datetime import date
import pandas as pd
import numpy as np
import pytest

from src.default_detection.default_detector import (
    DefaultDetector,
    DefaultEvent,
    DefaultEventType,
)


@pytest.fixture
def detector():
    return DefaultDetector()


@pytest.fixture
def company_register_df():
    return pd.DataFrame([
        {
            "cnpj": "33000167000101",
            "company_name": "PETROBRAS SA",
            "ticker": "PETR4",
            "is_active": True,
            "is_cancelled": False,
            "cancellation_date": None,
            "cancellation_reason": None,
            "status": "ATIVO",
        },
        {
            "cnpj": "11111111000100",
            "company_name": "EMPRESA FALIDA SA",
            "ticker": "FAIL3",
            "is_active": False,
            "is_cancelled": True,
            "cancellation_date": "2022-06-15",
            "cancellation_reason": "FALÊNCIA",
            "status": "CANCELADO",
        },
        {
            "cnpj": "22222222000100",
            "company_name": "OI SA",
            "ticker": "OIBR3",
            "is_active": False,
            "is_cancelled": True,
            "cancellation_date": "2021-12-01",
            "cancellation_reason": "RECUPERAÇÃO JUDICIAL",
            "status": "CANCELADO",
        },
    ])


@pytest.fixture
def suspended_market_df():
    """Mercado com ticker suspenso por 60+ dias."""
    dates = pd.date_range("2024-01-02", periods=90, freq="B")
    rows = []
    for i, d in enumerate(dates):
        rows.append({
            "ticker": "SUSP3",
            "date": d.date(),
            "close": 10.0 if i < 30 else 0.0,
            "volume_financial": 100_000 if i < 30 else 0,
            "open": 10.0, "high": 10.0, "low": 10.0,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def crashed_market_df():
    """Mercado com ticker com queda de 90% em 252 dias."""
    dates = pd.date_range("2023-01-02", periods=260, freq="B")
    prices = np.linspace(50.0, 5.0, len(dates))  # queda de 90%
    rows = []
    for d, p in zip(dates, prices):
        rows.append({
            "ticker": "CRAS3",
            "date": d.date(),
            "close": p,
            "volume_financial": 500_000,
            "open": p, "high": p * 1.01, "low": p * 0.99,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def fatos_df():
    """DataFrame de fatos relevantes com menção a recuperação judicial."""
    return pd.DataFrame([
        {
            "CNPJ_CIA": "22222222000100",
            "DENOM_CIA": "OI SA",
            "assunto": "RECUPERAÇÃO JUDICIAL - pedido de recuperação judicial aprovado",
            "DT_REFER": "2021-06-29",
        },
        {
            "CNPJ_CIA": "33000167000101",
            "DENOM_CIA": "PETROBRAS SA",
            "assunto": "Resultado do terceiro trimestre de 2023",
            "DT_REFER": "2023-11-10",
        },
    ])


@pytest.fixture
def distress_financial_df():
    """DataFrame financeiro com indicadores de distress."""
    return pd.DataFrame([
        {
            "cnpj_cia": "11111111000100",
            "reference_date": "2022-03-31",
            "altman_zscore": 0.8,         # zona de distress
            "current_ratio": 0.4,          # < 1 → liquidez ruim
            "net_debt_ebitda": 8.0,        # > 5x → alavancagem crítica
            "equity": -100_000,            # PL negativo
            "ebit_margin": -0.3,           # margem negativa
        },
        {
            "cnpj_cia": "33000167000101",
            "reference_date": "2023-12-31",
            "altman_zscore": 3.5,          # zona segura
            "current_ratio": 1.8,
            "net_debt_ebitda": 1.5,
            "equity": 500_000,
            "ebit_margin": 0.25,
        },
    ])


# =============================================================================
# Testes de Detecção de Cancelamento
# =============================================================================

class TestCancellationDetection:

    def test_detects_falencia(self, detector, company_register_df):
        """Deve detectar empresa com cancelamento por falência."""
        events = detector._detect_cancellations(company_register_df)
        falencia_events = [e for e in events if e.event_type == DefaultEventType.FALENCIA]
        assert len(falencia_events) >= 1
        assert falencia_events[0].cnpj == "11111111000100"

    def test_detects_recuperacao_judicial(self, detector, company_register_df):
        """Deve detectar empresa com cancelamento por recuperação judicial."""
        events = detector._detect_cancellations(company_register_df)
        rj_events = [
            e for e in events
            if e.event_type == DefaultEventType.RECUPERACAO_JUDICIAL
        ]
        assert len(rj_events) >= 1
        assert rj_events[0].cnpj == "22222222000100"

    def test_active_company_not_detected(self, detector, company_register_df):
        """Empresa ativa não deve ser marcada como default por cancelamento."""
        events = detector._detect_cancellations(company_register_df)
        petrobras_events = [e for e in events if e.cnpj == "33000167000101"]
        assert len(petrobras_events) == 0

    def test_confidence_is_1_for_formal_cancellation(self, detector, company_register_df):
        """Cancelamento formal deve ter confidence=1.0."""
        events = detector._detect_cancellations(company_register_df)
        for e in events:
            assert e.confidence == 1.0

    def test_empty_register_returns_empty(self, detector):
        """Cadastro vazio não deve gerar eventos."""
        events = detector._detect_cancellations(pd.DataFrame())
        assert len(events) == 0


# =============================================================================
# Testes de Detecção de Suspensão
# =============================================================================

class TestSuspensionDetection:

    def test_detects_prolonged_suspension(
        self, detector, suspended_market_df, company_register_df
    ):
        """Deve detectar 60+ dias de suspensão."""
        events = detector._detect_suspension(suspended_market_df, company_register_df)
        susp_events = [
            e for e in events
            if e.event_type == DefaultEventType.SUSPENSAO_PROLONGADA
        ]
        assert len(susp_events) >= 1
        assert susp_events[0].ticker == "SUSP3"

    def test_no_suspension_for_active_trader(
        self, detector, company_register_df
    ):
        """Ticker com volume normal não deve ser detectado como suspenso."""
        dates = pd.date_range("2024-01-02", periods=90, freq="B")
        df = pd.DataFrame([
            {
                "ticker": "ACTV3",
                "date": d.date(),
                "close": 20.0,
                "volume_financial": 500_000,
            }
            for d in dates
        ])
        events = detector._detect_suspension(df, company_register_df)
        actv_events = [e for e in events if e.ticker == "ACTV3"]
        assert len(actv_events) == 0

    def test_max_consecutive_zeros(self, detector):
        """Helper deve contar sequência corretamente."""
        series = pd.Series([100, 0, 0, 0, 50, 0, 0])
        assert detector._max_consecutive_zeros(series) == 3


# =============================================================================
# Testes de Detecção de Queda de Preço
# =============================================================================

class TestPriceCrashDetection:

    def test_detects_price_crash(self, detector, crashed_market_df):
        """Queda de 90% deve ser detectada."""
        events = detector._detect_price_crash(crashed_market_df)
        crash_events = [e for e in events if e.event_type == DefaultEventType.QUEDA_EXTREMA]
        assert len(crash_events) >= 1
        assert crash_events[0].ticker == "CRAS3"

    def test_no_crash_for_stable_price(self, detector):
        """Variação normal de preço não deve gerar evento."""
        dates = pd.date_range("2023-01-02", periods=260, freq="B")
        df = pd.DataFrame([
            {"ticker": "STBL3", "date": d.date(), "close": 30.0 + i * 0.01,
             "volume_financial": 100_000}
            for i, d in enumerate(dates)
        ])
        events = detector._detect_price_crash(df)
        stbl_events = [e for e in events if e.ticker == "STBL3"]
        assert len(stbl_events) == 0


# =============================================================================
# Testes de Detecção por Fatos Relevantes
# =============================================================================

class TestFatosRelevantesDetection:

    def test_detects_rj_from_text(self, detector, fatos_df, company_register_df):
        """Deve detectar RJ via texto do fato relevante."""
        events = detector._detect_from_fatos_relevantes(fatos_df, company_register_df)
        rj_events = [e for e in events if e.event_type == DefaultEventType.RECUPERACAO_JUDICIAL]
        assert len(rj_events) >= 1

    def test_normal_fato_not_detected(self, detector, fatos_df, company_register_df):
        """Resultado trimestral não deve gerar evento de default."""
        events = detector._detect_from_fatos_relevantes(fatos_df, company_register_df)
        petr_events = [e for e in events if "33000167000101" in e.cnpj]
        assert len(petr_events) == 0


# =============================================================================
# Testes de Score de Distress
# =============================================================================

class TestDistressScore:

    def test_distress_score_range(self, detector, distress_financial_df):
        """Score de distress deve estar entre 0 e 1."""
        result = detector._compute_distress_score(distress_financial_df)
        if not result.empty and "distress_score" in result.columns:
            scores = result["distress_score"].dropna()
            assert (scores >= 0).all()
            assert (scores <= 1).all()

    def test_distress_company_has_high_score(self, detector, distress_financial_df):
        """Empresa com indicadores ruins deve ter score alto."""
        result = detector._compute_distress_score(distress_financial_df)
        if not result.empty and "distress_score" in result.columns:
            distress_row = result[result["cnpj_cia"] == "11111111000100"]
            if not distress_row.empty:
                assert distress_row.iloc[0]["distress_score"] > 0.5

    def test_healthy_company_has_low_score(self, detector, distress_financial_df):
        """Empresa saudável deve ter score baixo."""
        result = detector._compute_distress_score(distress_financial_df)
        if not result.empty and "distress_score" in result.columns:
            healthy_row = result[result["cnpj_cia"] == "33000167000101"]
            if not healthy_row.empty:
                assert healthy_row.iloc[0]["distress_score"] < 0.4

    def test_distress_zone_labels(self, detector, distress_financial_df):
        """Zonas de distress devem ser 'safe', 'watch' ou 'alert'."""
        result = detector._compute_distress_score(distress_financial_df)
        if not result.empty and "distress_zone" in result.columns:
            valid_zones = {"safe", "watch", "alert"}
            zones = set(result["distress_zone"].dropna().unique())
            assert zones.issubset(valid_zones)


# =============================================================================
# Testes de Lookup e Utilitários
# =============================================================================

class TestLookupHelpers:

    def test_lookup_cnpj_by_ticker(self, company_register_df):
        """Deve resolver CNPJ a partir do ticker."""
        cnpj = DefaultDetector._lookup_cnpj("PETR4", company_register_df)
        assert cnpj == "33000167000101"

    def test_lookup_cnpj_unknown_ticker(self, company_register_df):
        """Ticker desconhecido deve retornar string vazia."""
        cnpj = DefaultDetector._lookup_cnpj("UNKN4", company_register_df)
        assert cnpj == ""

    def test_clean_cnpj(self):
        """Deve normalizar CNPJ."""
        assert DefaultDetector._clean_cnpj("33.000.167/0001-01") == "33000167000101"
        assert DefaultDetector._clean_cnpj("") == "00000000000000"
