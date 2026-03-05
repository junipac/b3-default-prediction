"""
Configuração global de testes — fixtures compartilhadas e configuração de pytest.
"""

import sys
from pathlib import Path

# Garante que o root do projeto está no PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np


@pytest.fixture(scope="session")
def sample_company_cnpjs():
    """CNPJs de empresas para testes (dados fictícios)."""
    return {
        "petrobras": "33000167000101",
        "vale": "33592510000154",
        "empresa_falida": "11111111000100",
        "empresa_rj": "22222222000100",
    }


@pytest.fixture(scope="session")
def full_analytical_df(sample_company_cnpjs):
    """Dataset analítico completo para testes de integração."""
    records = []
    for name, cnpj in sample_company_cnpjs.items():
        is_healthy = name in ["petrobras", "vale"]
        records.append({
            "cnpj_cia": cnpj,
            "reference_date": "2023-12-31",
            "company_name": name.upper() + " SA",
            "total_assets": 1_000_000 if is_healthy else 200_000,
            "total_liabilities_equity": 1_000_000 if is_healthy else 200_000,
            "equity": 400_000 if is_healthy else -50_000,
            "current_assets": 300_000 if is_healthy else 50_000,
            "current_liabilities": 150_000 if is_healthy else 200_000,
            "net_revenue": 800_000 if is_healthy else 100_000,
            "net_income": 100_000 if is_healthy else -80_000,
            "ebit": 150_000 if is_healthy else -60_000,
            "ebitda_approx": 200_000 if is_healthy else -40_000,
            "total_debt": 200_000 if is_healthy else 500_000,
            "net_debt": 100_000 if is_healthy else 480_000,
            "current_ratio": 2.0 if is_healthy else 0.25,
            "debt_to_equity": 0.5 if is_healthy else -10.0,
            "net_debt_ebitda": 0.5 if is_healthy else -12.0,
            "ebitda_margin": 0.25 if is_healthy else -0.4,
            "net_margin": 0.125 if is_healthy else -0.8,
            "roe": 0.25 if is_healthy else None,
            "roa": 0.10 if is_healthy else -0.4,
            "altman_zscore": 3.5 if is_healthy else 0.6,
            "altman_zone": "safe" if is_healthy else "distress",
            "retained_earnings": 200_000 if is_healthy else -300_000,
            "bs_check_pct": 0.0,
            "bs_balanced": True,
            "completeness_score": 0.95 if is_healthy else 0.60,
            "missing_pct": 0.05 if is_healthy else 0.40,
        })

    return pd.DataFrame(records)
