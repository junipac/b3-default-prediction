#!/usr/bin/env python3
"""
Gerador de Relatório Analítico Avançado - Modelo de Merton.

Conecta a dados reais da CVM, calcula PD estrutural (Merton),
índice de saúde setorial (SHI), e gera relatório HTML completo.

Saídas:
    - relatorio_merton_b3.html (relatório visual completo)
    - merton_dataset.csv (dataset completo com PDs)
    - merton_por_setor.csv (agregação setorial)
    - merton_validacao.csv (resultados de validação)
"""

import os
import sys
import io
import zipfile
import hashlib
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# Adicionar raiz ao path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.models.merton_model import MertonModel, MertonInput, DriftMode
from src.analytics.sector_monitor import SectorMonitor, WeightingMethod
from src.analytics.health_index import SectoralHealthIndex
from src.analytics.pd_aggregation import PDAggregation
from src.analytics.visualization import (
    plot_pd_by_sector, plot_dd_distribution, plot_pd_distribution,
    plot_sector_heatmap, plot_shi_chart, plot_rating_distribution,
    plot_merton_scatter, plot_convergence_stats,
    plot_temporal_pd_variation, plot_temporal_indicators,
    plot_sector_migration,
)
from src.analytics.validation import ModelValidator


# ============================================================
# 1. EXTRAÇÃO DE DADOS REAIS DA CVM
# ============================================================

CVM_BASE = "https://dados.cvm.gov.br/dados/CIA_ABERTA"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# Taxa SELIC (constante global)
SELIC = 0.1375  # 13.75% a.a.

# Setores financeiros (padrões para matching parcial case-insensitive)
# Esses setores possuem alavancagem estrutural elevada (depósitos, recursos de terceiros)
# que não representam risco de default no sentido do modelo de Merton.
FINANCIAL_SECTOR_PATTERNS = [
    'banco', 'bancos',
    'financeira', 'financeiro', 'financeiras',
    'seguradora', 'seguradoras', 'seguros',
    'corretora', 'corretoras',
    'securitiz',           # Securitização de Recebíveis, etc.
    'crédito',             # Sociedade de Crédito
    'previdência',         # Previdência e Seguros
    'arrendamento',        # Arrendamento Mercantil
    'factoring',
    'holding financeira',
    'bolsa de valores',
    'intermediação financeira',
    'serviços financeiros',
    'emp. adm. part.',     # Holdings (estruturalmente alavancadas como financeiras)
]

# Fator de ajuste da barreira de default para empresas financeiras
# Fundamentação: grande parte do passivo de instituições financeiras é composto por
# depósitos, recursos interbancários e captações que não representam dívida "distress"
# no sentido estrutural de Merton. O fator reduz a barreira efetiva para refletir
# apenas a parcela de dívida sensível a mercado (market-sensitive debt).
FINANCIAL_DEBT_BARRIER_FACTOR = 0.20  # usa apenas 20% da dívida como barreira

# Cap máximo de D/E efetivo para empresas financeiras após ajuste.
# Mesmo com fator de 20%, D/E original de 300x geraria D/E ajustado de 60x,
# ainda distorcendo o modelo. O cap garante que a barreira de dívida efetiva
# não exceda MAX_FINANCIAL_DE × equity, produzindo PDs razoáveis.
MAX_FINANCIAL_DE = 3.0


def is_financial_sector(sector_name) -> bool:
    """Verifica se o setor é do segmento financeiro."""
    if pd.isna(sector_name) or not sector_name:
        return False
    sector_lower = str(sector_name).lower().strip()
    for pattern in FINANCIAL_SECTOR_PATTERNS:
        if pattern in sector_lower:
            return True
    return False


def adjust_financial_debt(inp) -> None:
    """Ajusta barreira de dívida para empresa financeira.

    Aplica dois mecanismos:
      1) Fator percentual: reduz dívida a FINANCIAL_DEBT_BARRIER_FACTOR (20%)
      2) Cap absoluto: limita dívida efetiva a MAX_FINANCIAL_DE × equity

    O mais restritivo dos dois prevalece, evitando que D/E extremos
    (ex.: XP com D/E=309x) ainda distorçam o modelo após o fator linear.
    """
    # Passo 1: fator percentual
    adj_st = inp.short_term_debt * FINANCIAL_DEBT_BARRIER_FACTOR
    adj_lt = inp.long_term_debt * FINANCIAL_DEBT_BARRIER_FACTOR

    # Passo 2: cap por equity (D_barrier = ST + 0.5*LT deve ser ≤ MAX_DE * E)
    equity = inp.equity_value
    if equity > 0:
        barrier_after_factor = adj_st + 0.5 * adj_lt
        max_barrier = MAX_FINANCIAL_DE * equity
        if barrier_after_factor > max_barrier and barrier_after_factor > 0:
            # Escala proporcional para respeitar o cap
            scale = max_barrier / barrier_after_factor
            adj_st *= scale
            adj_lt *= scale

    inp.short_term_debt = adj_st
    inp.long_term_debt = adj_lt

# Mapeamento CD_CONTA → nome padronizado
ACCOUNT_MAP = {
    "1": "total_assets",
    "1.01": "current_assets",
    "1.01.01": "cash_equivalents",
    "1.01.03": "receivables",
    "1.01.04": "inventories",
    "1.02": "non_current_assets",
    "1.02.04": "ppe",
    "2": "total_liabilities_equity",
    "2.01": "current_liabilities",
    "2.01.04": "short_term_debt",
    "2.02": "non_current_liabilities",
    "2.02.01": "long_term_debt",
    "2.03": "equity",
    "2.03.01": "paid_in_capital",
    "2.03.09": "retained_earnings",
    "3.01": "net_revenue",
    "3.02": "cogs",
    "3.03": "gross_profit",
    "3.05": "ebit",
    "3.06": "financial_result",
    "3.07": "ebt",
    "3.09": "net_income_before_minority",
    "3.11": "net_income",
    "6.01": "cfo",
    "6.02": "cfi",
    "6.03": "cff",
}


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def fetch_url(url: str, timeout: int = 120) -> requests.Response:
    """GET com retry simples."""
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            if attempt < 2:
                import time
                time.sleep(2 ** attempt)
            else:
                raise
    raise RuntimeError(f"Falha ao acessar {url}")


def read_csv_robust(raw_bytes: bytes, filename: str = "") -> pd.DataFrame:
    """Lê CSV com múltiplos encodings e separadores."""
    encodings = ['latin-1', 'utf-8', 'cp1252', 'iso-8859-1']
    separators = [';', ',', '|']

    for enc in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(
                    io.BytesIO(raw_bytes),
                    sep=sep, encoding=enc,
                    low_memory=False, dtype=str,
                )
                if len(df.columns) >= 5 and len(df) > 0:
                    return df
            except Exception:
                continue
    return pd.DataFrame()


def fetch_company_register() -> pd.DataFrame:
    """Baixa cadastro de empresas da CVM."""
    log("Baixando cadastro de empresas CVM...")
    url = f"{CVM_BASE}/CAD/DADOS/cad_cia_aberta.csv"
    resp = fetch_url(url)
    df = read_csv_robust(resp.content, "cad_cia_aberta.csv")
    log(f"  → {len(df)} empresas no cadastro")
    return df


def fetch_dfp_year(year: int) -> dict:
    """Baixa DFP (demonstrações financeiras) de um ano."""
    log(f"Baixando DFP {year}...")
    url = f"{CVM_BASE}/DOC/DFP/DADOS/dfp_cia_aberta_{year}.zip"
    resp = fetch_url(url)

    dfs = {}
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for name in zf.namelist():
            if name.endswith('.csv'):
                raw = zf.read(name)
                df = read_csv_robust(raw, name)
                if not df.empty:
                    key = name.lower().replace(f'dfp_cia_aberta_', '').replace(
                        f'_{year}.csv', '').replace('.csv', '')
                    # Limpar chave
                    for prefix in ['con_', 'ind_']:
                        if key.startswith(prefix):
                            key = key[len(prefix):]
                    dfs[name.lower()] = df
                    log(f"  → {name}: {len(df)} linhas")

    return dfs


def normalize_financials(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Normaliza DataFrame financeiro da CVM."""
    if df.empty:
        return df

    df = df.copy()

    # Padronizar nomes de colunas
    col_map = {}
    for col in df.columns:
        upper = col.upper().strip()
        if upper in ('CNPJ_CIA', 'CNPJ'):
            col_map[col] = 'CNPJ_CIA'
        elif upper == 'DENOM_CIA':
            col_map[col] = 'DENOM_CIA'
        elif upper == 'CD_CONTA':
            col_map[col] = 'CD_CONTA'
        elif upper == 'DS_CONTA':
            col_map[col] = 'DS_CONTA'
        elif upper == 'VL_CONTA':
            col_map[col] = 'VL_CONTA'
        elif upper == 'DT_REFER':
            col_map[col] = 'DT_REFER'
        elif upper == 'DT_INI_EXERC':
            col_map[col] = 'DT_INI_EXERC'
        elif upper == 'DT_FIM_EXERC':
            col_map[col] = 'DT_FIM_EXERC'
        elif upper == 'ORDEM_EXERC':
            col_map[col] = 'ORDEM_EXERC'
        elif upper == 'ESCALA_MOEDA':
            col_map[col] = 'ESCALA_MOEDA'
        elif upper == 'VERSAO':
            col_map[col] = 'VERSAO'
        elif upper == 'CD_CVM':
            col_map[col] = 'CD_CVM'

    df = df.rename(columns=col_map)

    required = ['CNPJ_CIA', 'CD_CONTA', 'VL_CONTA']
    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    # Filtrar período corrente
    if 'ORDEM_EXERC' in df.columns:
        mask = df['ORDEM_EXERC'].astype(str).str.upper().str.contains('LTIMO|ULTIMO|LAST', na=False)
        if mask.any():
            df = df[mask]

    # Converter valor
    df['VL_CONTA'] = (
        df['VL_CONTA'].astype(str)
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False)
    )
    df['VL_CONTA'] = pd.to_numeric(df['VL_CONTA'], errors='coerce')

    # Escala
    if 'ESCALA_MOEDA' in df.columns:
        scale_map = {'MIL': 1000, 'UNIDADE': 1, 'MILHAO': 1e6, 'BILHAO': 1e9}
        df['_scale'] = df['ESCALA_MOEDA'].str.upper().str.strip().map(scale_map).fillna(1)
        df['VL_CONTA'] = df['VL_CONTA'] * df['_scale']
        df = df.drop(columns=['_scale'])

    # Versão mais recente
    if 'VERSAO' in df.columns:
        df['VERSAO'] = pd.to_numeric(df['VERSAO'], errors='coerce').fillna(1)
        df = df.sort_values('VERSAO', ascending=False)
        df = df.drop_duplicates(subset=['CNPJ_CIA', 'CD_CONTA'], keep='first')

    # Mapear contas
    df['account_standard'] = df['CD_CONTA'].map(ACCOUNT_MAP)

    df['source_year'] = year

    return df


def pivot_to_analytical(frames: dict) -> pd.DataFrame:
    """Pivota demonstrações financeiras em dataset analítico."""
    all_data = []

    for key, df in frames.items():
        if df.empty:
            continue
        mapped = df[df['account_standard'].notna()].copy()
        if not mapped.empty:
            all_data.append(mapped)

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)

    # Deduplicar por empresa + conta (manter maior versão)
    combined = combined.sort_values('source_year', ascending=False)
    combined = combined.drop_duplicates(
        subset=['CNPJ_CIA', 'account_standard', 'source_year'], keep='first'
    )

    # Pivot: uma linha por empresa
    pivot = combined.pivot_table(
        index=['CNPJ_CIA', 'DENOM_CIA', 'source_year'],
        columns='account_standard',
        values='VL_CONTA',
        aggfunc='first',
    ).reset_index()

    # Flatten MultiIndex
    pivot.columns = [
        col if isinstance(col, str) else col
        for col in pivot.columns
    ]

    return pivot


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula indicadores financeiros derivados."""
    df = df.copy()

    # Indicadores de liquidez
    ca = df.get('current_assets', pd.Series(dtype=float))
    cl = df.get('current_liabilities', pd.Series(dtype=float))
    inv = df.get('inventories', pd.Series(dtype=float)).fillna(0)
    cash = df.get('cash_equivalents', pd.Series(dtype=float)).fillna(0)

    df['current_ratio'] = np.where(cl > 0, ca / cl, np.nan)
    df['quick_ratio'] = np.where(cl > 0, (ca - inv) / cl, np.nan)
    df['cash_ratio'] = np.where(cl > 0, cash / cl, np.nan)

    # Dívida
    st = df.get('short_term_debt', pd.Series(dtype=float)).fillna(0)
    lt = df.get('long_term_debt', pd.Series(dtype=float)).fillna(0)
    df['total_debt'] = st + lt
    eq = df.get('equity', pd.Series(dtype=float))
    df['debt_to_equity'] = np.where(
        (eq.notna()) & (eq != 0), df['total_debt'] / eq, np.nan
    )
    df['net_debt'] = df['total_debt'] - cash

    # Rentabilidade
    ta = df.get('total_assets', pd.Series(dtype=float))
    ni = df.get('net_income', pd.Series(dtype=float))
    rev = df.get('net_revenue', pd.Series(dtype=float))
    ebit = df.get('ebit', pd.Series(dtype=float))
    gp = df.get('gross_profit', pd.Series(dtype=float))

    df['roa'] = np.where((ta.notna()) & (ta != 0), ni / ta, np.nan)
    df['roe'] = np.where((eq.notna()) & (eq != 0), ni / eq, np.nan)

    # Margens
    df['gross_margin'] = np.where(
        (rev.notna()) & (rev != 0), gp / rev, np.nan
    )
    df['ebit_margin'] = np.where(
        (rev.notna()) & (rev != 0), ebit / rev, np.nan
    )
    df['net_margin'] = np.where(
        (rev.notna()) & (rev != 0), ni / rev, np.nan
    )

    # EBITDA aproximado
    ppe = df.get('ppe', pd.Series(dtype=float)).fillna(0)
    intang = df.get('intangibles', pd.Series(dtype=float)).fillna(0)
    deprec_proxy = (ppe + intang) * 0.1  # proxy 10% de imob+intang
    df['ebitda_approx'] = ebit.fillna(0) + deprec_proxy

    # Cobertura de juros
    fin_result = df.get('financial_result', pd.Series(dtype=float)).fillna(0)
    interest_exp = fin_result.clip(upper=0).abs()
    df['interest_coverage'] = np.where(
        interest_exp > 0, ebit / interest_exp, np.nan
    )

    # Altman Z-Score (mercados emergentes)
    wc = ca.fillna(0) - cl.fillna(0)
    re = df.get('retained_earnings', pd.Series(dtype=float)).fillna(0)
    tl = df.get('total_liabilities_equity', pd.Series(dtype=float)).fillna(0) - eq.fillna(0)

    x1 = np.where(ta > 0, wc / ta, 0)
    x2 = np.where(ta > 0, re / ta, 0)
    x3 = np.where(ta > 0, ebit.fillna(0) / ta, 0)
    x4 = np.where(tl > 0, eq.fillna(0) / tl, 0)

    df['altman_zscore'] = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4
    df['altman_zone'] = pd.cut(
        df['altman_zscore'],
        bins=[-np.inf, 1.1, 2.6, np.inf],
        labels=['Distress', 'Grey', 'Safe'],
    )

    return df


# ============================================================
# 2. ANÁLISE TEMPORAL
# ============================================================

def build_temporal_comparison(
    analytical_all: pd.DataFrame,
    reg_slim: pd.DataFrame,
    merton: MertonModel,
    vol_map: dict,
) -> pd.DataFrame:
    """
    Constrói comparação temporal por setor entre 2023 e 2024.

    Para cada ano, calcula PD Merton por empresa e agrega por setor.
    Retorna DataFrame com delta de indicadores entre períodos.
    """
    if analytical_all.empty or 'source_year' not in analytical_all.columns:
        return pd.DataFrame()

    years = sorted(analytical_all['source_year'].unique())
    if len(years) < 2:
        return pd.DataFrame()

    year_old, year_new = years[0], years[-1]

    def get_sector_vol(sector_name):
        if pd.isna(sector_name):
            return 0.45
        for key, vol in vol_map.items():
            if key.lower() in str(sector_name).lower():
                return vol
        return 0.45

    year_results = {}

    for year in [year_old, year_new]:
        year_df = analytical_all[analytical_all['source_year'] == year].copy()
        year_df = year_df.drop_duplicates(subset=['CNPJ_CIA'], keep='first')

        if year_df.empty:
            continue

        # Volatilidade
        if 'sector' in year_df.columns:
            year_df['equity_volatility'] = year_df['sector'].apply(get_sector_vol)
        else:
            year_df['equity_volatility'] = 0.45

        year_df['cnpj_cia'] = year_df['CNPJ_CIA']
        year_df['company_name'] = year_df['DENOM_CIA']
        year_df['reference_date'] = str(year)

        inputs = MertonModel.prepare_inputs_from_fundamentals(
            year_df, risk_free_rate=SELIC,
            drift_mode=DriftMode.RISK_NEUTRAL, equity_vol_default=0.45,
        )

        if not inputs:
            continue

        # Ajuste financeiro também na análise temporal
        for inp in inputs:
            if is_financial_sector(inp.sector):
                adjust_financial_debt(inp)

        merton_df = merton.compute_batch(inputs)
        if merton_df.empty:
            continue

        # Merge para ter setor
        merton_df = merton_df.rename(columns={'cnpj': 'CNPJ_CIA'})
        merged = year_df.merge(
            merton_df[['CNPJ_CIA', 'pd_merton', 'distance_to_default']],
            on='CNPJ_CIA', how='inner', suffixes=('', '_merton')
        )

        # Agregar por setor
        sector_col = 'sector' if 'sector' in merged.columns else None
        if not sector_col:
            continue

        sector_agg = merged.groupby(sector_col).agg(
            n_empresas=('CNPJ_CIA', 'count'),
            pd_media=('pd_merton', 'mean'),
            pd_mediana=('pd_merton', 'median'),
            dd_media=('distance_to_default', 'median'),
            roa_media=('roa', 'median') if 'roa' in merged.columns else ('CNPJ_CIA', 'count'),
            leverage_media=('debt_to_equity', 'median') if 'debt_to_equity' in merged.columns else ('CNPJ_CIA', 'count'),
            altman_media=('altman_zscore', 'median') if 'altman_zscore' in merged.columns else ('CNPJ_CIA', 'count'),
            current_ratio_media=('current_ratio', 'median') if 'current_ratio' in merged.columns else ('CNPJ_CIA', 'count'),
            ebit_margin_media=('ebit_margin', 'median') if 'ebit_margin' in merged.columns else ('CNPJ_CIA', 'count'),
        ).reset_index()
        sector_agg = sector_agg.rename(columns={sector_col: 'sector'})

        year_results[year] = sector_agg

    if len(year_results) < 2:
        return pd.DataFrame()

    old_df = year_results[year_old].add_suffix(f'_{year_old}')
    old_df = old_df.rename(columns={f'sector_{year_old}': 'sector'})
    new_df = year_results[year_new].add_suffix(f'_{year_new}')
    new_df = new_df.rename(columns={f'sector_{year_new}': 'sector'})

    temporal = old_df.merge(new_df, on='sector', how='inner')

    # Calcular deltas
    temporal['delta_pd'] = (
        temporal[f'pd_media_{year_new}'] - temporal[f'pd_media_{year_old}']
    )
    temporal['delta_dd'] = (
        temporal[f'dd_media_{year_new}'] - temporal[f'dd_media_{year_old}']
    )

    # Renomear para compatibilidade com visualizações
    temporal['pd_2023'] = temporal[f'pd_media_{year_old}']
    temporal['pd_2024'] = temporal[f'pd_media_{year_new}']
    temporal['n_common'] = temporal[[
        f'n_empresas_{year_old}', f'n_empresas_{year_new}'
    ]].min(axis=1)

    # Deltas de indicadores (se colunas são numéricas e não de count)
    for indicator, col_old, col_new in [
        ('delta_roa', f'roa_media_{year_old}', f'roa_media_{year_new}'),
        ('delta_leverage', f'leverage_media_{year_old}', f'leverage_media_{year_new}'),
        ('delta_altman', f'altman_media_{year_old}', f'altman_media_{year_new}'),
    ]:
        if col_old in temporal.columns and col_new in temporal.columns:
            old_vals = pd.to_numeric(temporal[col_old], errors='coerce')
            new_vals = pd.to_numeric(temporal[col_new], errors='coerce')
            temporal[indicator] = new_vals - old_vals

    # Classificação
    temporal['trend'] = temporal['delta_pd'].apply(
        lambda d: 'Melhoria' if d < -0.005 else ('Deterioração' if d > 0.005 else 'Estável')
    )

    temporal = temporal.sort_values('delta_pd')

    return temporal


# ============================================================
# 3. PIPELINE PRINCIPAL
# ============================================================

def run_pipeline():
    """Executa pipeline completo: extração → Merton → relatório."""

    output_dir = ROOT / "data" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 2.1 Extração de dados
    # --------------------------------------------------------
    log("=" * 60)
    log("PIPELINE DE ANÁLISE AVANÇADA - MODELO DE MERTON")
    log("=" * 60)

    register = fetch_company_register()

    # Baixar DFP 2023 e 2024
    all_frames = {}
    for year in [2023, 2024]:
        try:
            year_dfs = fetch_dfp_year(year)
            for key, df in year_dfs.items():
                norm = normalize_financials(df, year)
                if not norm.empty and 'account_standard' in norm.columns:
                    if key not in all_frames:
                        all_frames[key] = []
                    all_frames[key].append(norm)
        except Exception as e:
            log(f"  ⚠ Erro DFP {year}: {e}")

    # Consolidar frames por tipo
    consolidated = {}
    for key, frame_list in all_frames.items():
        consolidated[key] = pd.concat(frame_list, ignore_index=True)

    # Pivot para dataset analítico
    log("\nConstruindo dataset analítico...")
    analytical = pivot_to_analytical(consolidated)

    if analytical.empty:
        log("ERRO: Dataset analítico vazio!")
        return

    log(f"  → {len(analytical)} registros no dataset analítico")

    # Calcular indicadores
    log("Calculando indicadores financeiros...")
    analytical = compute_indicators(analytical)

    # Merge com setor do cadastro
    reg_slim = None
    if not register.empty:
        sector_col = None
        for c in register.columns:
            if c.upper() == 'SETOR_ATIV':
                sector_col = c
                break

        cnpj_col_reg = None
        for c in register.columns:
            if c.upper() == 'CNPJ_CIA':
                cnpj_col_reg = c
                break

        if sector_col and cnpj_col_reg:
            reg_slim = register[[cnpj_col_reg, sector_col]].drop_duplicates(
                subset=[cnpj_col_reg], keep='first'
            )
            reg_slim = reg_slim.rename(columns={
                cnpj_col_reg: 'CNPJ_CIA',
                sector_col: 'sector',
            })
            analytical = analytical.merge(reg_slim, on='CNPJ_CIA', how='left')
            analytical['sector'] = analytical['sector'].fillna('Não Classificado')

    # Guardar dataset completo (ambos anos) para análise temporal
    analytical_all_years = analytical.copy()
    log(f"  → {len(analytical_all_years)} registros (todos os anos)")

    # Dataset principal: mais recente por empresa
    analytical = analytical.sort_values('source_year', ascending=False)
    analytical = analytical.drop_duplicates(subset=['CNPJ_CIA'], keep='first')
    log(f"  → {len(analytical)} empresas únicas (mais recente)")

    # --------------------------------------------------------
    # 2.2 Modelo de Merton
    # --------------------------------------------------------
    log("\n" + "=" * 60)
    log("MODELO ESTRUTURAL DE MERTON")
    log("=" * 60)

    # Preparar inputs
    log("Preparando inputs para Merton...")

    # Estimar volatilidade do equity por setor
    sector_vol = {
        'default': 0.45,
    }
    # Heurística: setores mais voláteis
    vol_map = {
        'Construção Civil': 0.55,
        'Comércio': 0.40,
        'Energia Elétrica': 0.35,
        'Telecomunicações': 0.40,
        'Siderurgia e Metalurgia': 0.50,
        'Petroquímicos e Borracha': 0.50,
        'Bancos': 0.35,
        'Seguradoras e Corretoras': 0.35,
        'Alimentos': 0.38,
        'Transporte Aéreo': 0.60,
        'Mineração': 0.55,
        'Têxtil e Vestuário': 0.50,
    }

    def get_sector_vol(sector_name):
        if pd.isna(sector_name):
            return 0.45
        for key, vol in vol_map.items():
            if key.lower() in str(sector_name).lower():
                return vol
        return 0.45

    analytical['equity_volatility'] = analytical['sector'].apply(get_sector_vol)

    # Renomear para compatibilidade com Merton
    analytical['cnpj_cia'] = analytical['CNPJ_CIA']
    analytical['company_name'] = analytical['DENOM_CIA']
    if 'DT_REFER' in analytical.columns:
        analytical['reference_date'] = analytical['DT_REFER']
    else:
        analytical['reference_date'] = str(datetime.now().date())

    # Construir inputs
    merton = MertonModel(max_iterations=200, tolerance=1e-8)
    inputs = MertonModel.prepare_inputs_from_fundamentals(
        analytical,
        risk_free_rate=SELIC,
        drift_mode=DriftMode.RISK_NEUTRAL,
        equity_vol_default=0.45,
    )

    # Ajuste para setor financeiro: reduzir barreira de default
    # Instituições financeiras (bancos, seguradoras, corretoras, holdings) possuem
    # alavancagem estrutural elevada que não representa risco de default no sentido
    # de Merton. Aplicamos fator + cap adaptativo de D/E.
    n_financial_adj = 0
    for inp in inputs:
        if is_financial_sector(inp.sector):
            adjust_financial_debt(inp)
            n_financial_adj += 1

    log(f"  → {len(inputs)} empresas com dados suficientes para Merton")
    log(f"  → {n_financial_adj} empresas financeiras com barreira de dívida ajustada (fator {FINANCIAL_DEBT_BARRIER_FACTOR:.0%}, cap D/E≤{MAX_FINANCIAL_DE:.0f}×)")

    # Calcular PD
    log("Resolvendo sistema não-linear de Merton...")
    merton_results = merton.compute_batch(inputs)

    if merton_results.empty:
        log("ERRO: Nenhum resultado de Merton!")
        return

    converged = merton_results['converged'].sum()
    total = len(merton_results)
    log(f"  → Convergência: {converged}/{total} ({converged/total*100:.1f}%)")
    log(f"  → PD média: {merton_results['pd_merton'].mean()*100:.2f}%")
    log(f"  → PD mediana: {merton_results['pd_merton'].median()*100:.2f}%")
    log(f"  → DD média: {merton_results['distance_to_default'].mean():.2f}")

    # Merge Merton com dataset analítico
    merton_merge = merton_results[[
        'cnpj', 'pd_merton', 'distance_to_default', 'asset_value',
        'asset_volatility', 'leverage_ratio', 'converged', 'iterations',
        'rating_bucket', 'd1', 'd2', 'drift',
    ]].copy()
    merton_merge = merton_merge.rename(columns={'cnpj': 'CNPJ_CIA'})

    dataset = analytical.merge(merton_merge, on='CNPJ_CIA', how='inner')

    # Marcar empresas financeiras no dataset
    dataset['is_financial'] = dataset['sector'].apply(is_financial_sector)
    n_fin = dataset['is_financial'].sum()
    pd_fin = dataset.loc[dataset['is_financial'], 'pd_merton'].mean() * 100
    pd_nonfin = dataset.loc[~dataset['is_financial'], 'pd_merton'].mean() * 100

    log(f"  → Dataset final: {len(dataset)} empresas com PD Merton")
    log(f"  → {n_fin} empresas financeiras (PD média: {pd_fin:.2f}%)")
    log(f"  → {len(dataset) - n_fin} empresas não-financeiras (PD média: {pd_nonfin:.2f}%)")

    # --------------------------------------------------------
    # 2.3 Análise Setorial
    # --------------------------------------------------------
    log("\n" + "=" * 60)
    log("ANÁLISE SETORIAL")
    log("=" * 60)

    # Sector Monitor
    monitor = SectorMonitor(weighting=WeightingMethod.EQUAL)
    sector_metrics = monitor.compute_sector_metrics(dataset, merton_results)
    log(f"  → {len(sector_metrics)} setores analisados")

    # SHI
    shi = SectoralHealthIndex()
    shi_results = shi.compute(dataset)
    log(f"  → SHI calculado para {len(shi_results)} setores")

    # PD Aggregation
    agg = PDAggregation()
    sector_pd = agg.aggregate_by_sector(
        dataset, pd_col='pd_merton', sector_col='sector',
        weight_col='total_assets',
    )
    rating_dist = agg.rating_distribution(dataset, pd_col='pd_merton')
    concentration = agg.compute_concentration_metrics(sector_pd)

    log(f"  → PD média geral: {sector_pd['pd_media'].mean()*100:.2f}%")
    log(f"  → HHI: {concentration.get('hhi', 0):.4f}")

    # --------------------------------------------------------
    # 2.4 Validação
    # --------------------------------------------------------
    log("\n" + "=" * 60)
    log("VALIDAÇÃO DO MODELO")
    log("=" * 60)

    validator = ModelValidator()
    backtest = validator.backtest(dataset, pd_col='pd_merton')
    log(f"  → N total: {backtest.n_total}")
    log(f"  → N defaults (proxy): {backtest.n_defaults}")
    log(f"  → KS: {backtest.ks_statistic:.4f}")
    log(f"  → Gini: {backtest.gini_coefficient:.4f}")
    log(f"  → PD ratio (def/non-def): {backtest.pd_ratio:.2f}x")

    sensitivity = validator.sensitivity_analysis(dataset, pd_col='pd_merton')

    correlation = validator.correlation_with_defaults(dataset, pd_col='pd_merton')
    log(f"  → Correlação rank PD-default: {correlation['rank_correlation']:.4f}")

    # --------------------------------------------------------
    # 2.5 Análise Temporal (2023 vs 2024)
    # --------------------------------------------------------
    log("\n" + "=" * 60)
    log("ANÁLISE TEMPORAL (2023 vs 2024)")
    log("=" * 60)

    temporal_df = build_temporal_comparison(
        analytical_all_years, reg_slim, merton, vol_map
    )
    if not temporal_df.empty:
        n_improved = (temporal_df['delta_pd'] < 0).sum()
        n_worsened = (temporal_df['delta_pd'] > 0).sum()
        log(f"  → {len(temporal_df)} setores com dados em ambos os anos")
        log(f"  → {n_improved} setores melhoraram, {n_worsened} pioraram")
        avg_delta = temporal_df['delta_pd'].mean() * 100
        log(f"  → Δ PD médio: {avg_delta:+.2f} pp")
    else:
        log("  → Dados temporais insuficientes")

    # --------------------------------------------------------
    # 2.6 Gerar Gráficos
    # --------------------------------------------------------
    log("\n" + "=" * 60)
    log("GERANDO VISUALIZAÇÕES")
    log("=" * 60)

    charts = {}

    charts['pd_sector'] = plot_pd_by_sector(sector_pd, top_n=20)
    log("  → Gráfico PD por setor")

    charts['dd_dist'] = plot_dd_distribution(dataset)
    log("  → Distribuição DD")

    charts['pd_dist'] = plot_pd_distribution(dataset)
    log("  → Distribuição PD")

    charts['heatmap'] = plot_sector_heatmap(sector_pd)
    log("  → Heatmap setorial")

    if not shi_results.empty:
        charts['shi'] = plot_shi_chart(shi_results)
        log("  → Gráfico SHI")

    charts['rating'] = plot_rating_distribution(rating_dist)
    log("  → Distribuição ratings")

    charts['scatter'] = plot_merton_scatter(dataset)
    log("  → Scatter leverage vs DD")

    charts['convergence'] = plot_convergence_stats(merton_results)
    log("  → Convergência solver")

    if not temporal_df.empty:
        charts['temporal_pd'] = plot_temporal_pd_variation(temporal_df)
        log("  → Variação temporal PD")
        charts['temporal_indicators'] = plot_temporal_indicators(temporal_df)
        log("  → Indicadores temporais")
        charts['migration'] = plot_sector_migration(temporal_df)
        log("  → Migração setorial")

    # --------------------------------------------------------
    # 2.7 Gerar Relatório HTML
    # --------------------------------------------------------
    log("\n" + "=" * 60)
    log("GERANDO RELATÓRIO HTML")
    log("=" * 60)

    html = generate_html(
        dataset=dataset,
        merton_results=merton_results,
        sector_pd=sector_pd,
        rating_dist=rating_dist,
        shi_results=shi_results,
        backtest=backtest,
        sensitivity=sensitivity,
        concentration=concentration,
        correlation=correlation,
        charts=charts,
        temporal_df=temporal_df,
    )

    # Salvar outputs
    report_path = output_dir / "relatorio_merton_b3.html"
    report_path.write_text(html, encoding='utf-8')
    log(f"  → {report_path} ({report_path.stat().st_size / 1024:.0f}K)")

    # CSVs
    csv_cols = [
        'CNPJ_CIA', 'DENOM_CIA', 'sector', 'source_year',
        'total_assets', 'equity', 'total_debt', 'net_revenue',
        'net_income', 'ebit', 'roa', 'roe', 'current_ratio',
        'debt_to_equity', 'altman_zscore', 'altman_zone',
        'pd_merton', 'distance_to_default', 'asset_value',
        'asset_volatility', 'leverage_ratio', 'rating_bucket',
        'converged', 'is_financial',
    ]
    csv_cols = [c for c in csv_cols if c in dataset.columns]
    dataset[csv_cols].to_csv(
        output_dir / "merton_dataset.csv", index=False, encoding='utf-8-sig'
    )

    sector_pd.to_csv(
        output_dir / "merton_por_setor.csv", index=False, encoding='utf-8-sig'
    )

    sensitivity.to_csv(
        output_dir / "merton_sensibilidade.csv", index=False, encoding='utf-8-sig'
    )

    if not temporal_df.empty:
        temporal_df.to_csv(
            output_dir / "merton_temporal.csv", index=False, encoding='utf-8-sig'
        )

    log("\n" + "=" * 60)
    log("PIPELINE CONCLUÍDO COM SUCESSO")
    log("=" * 60)

    return dataset, merton_results, sector_pd


# ============================================================
# 3. GERAÇÃO DO HTML
# ============================================================

def generate_html(
    dataset, merton_results, sector_pd, rating_dist,
    shi_results, backtest, sensitivity, concentration,
    correlation, charts, temporal_df=None,
) -> str:
    """Gera relatório HTML completo."""

    n_total = len(dataset)
    n_converged = merton_results['converged'].sum() if not merton_results.empty else 0
    pd_mean = dataset['pd_merton'].mean() * 100 if 'pd_merton' in dataset.columns else 0
    pd_median = dataset['pd_merton'].median() * 100 if 'pd_merton' in dataset.columns else 0
    dd_mean = dataset['distance_to_default'].mean() if 'distance_to_default' in dataset.columns else 0

    n_sectors = dataset['sector'].nunique() if 'sector' in dataset.columns else 0

    n_distress = (dataset.get('pd_merton', pd.Series()) > 0.15).sum()
    n_ig = dataset['rating_bucket'].isin({'AAA', 'AA', 'A', 'BBB'}).sum() if 'rating_bucket' in dataset.columns else 0

    # Top distress
    top_distress = dataset.nlargest(25, 'pd_merton')[
        [c for c in ['DENOM_CIA', 'sector', 'pd_merton', 'distance_to_default',
                      'altman_zscore', 'debt_to_equity', 'roa', 'rating_bucket',
                      'total_assets'] if c in dataset.columns]
    ] if 'pd_merton' in dataset.columns else pd.DataFrame()

    # Top safe
    top_safe = dataset.nsmallest(20, 'pd_merton')[
        [c for c in ['DENOM_CIA', 'sector', 'pd_merton', 'distance_to_default',
                      'altman_zscore', 'current_ratio', 'roa', 'rating_bucket']
         if c in dataset.columns]
    ] if 'pd_merton' in dataset.columns else pd.DataFrame()

    timestamp = datetime.now().strftime('%d/%m/%Y %H:%M')

    html = f'''<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Relatório PD Merton - B3 | {timestamp}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f7fa; color: #333; line-height: 1.6; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
header {{ background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%); color: white; padding: 40px; border-radius: 12px; margin-bottom: 30px; }}
header h1 {{ font-size: 28px; margin-bottom: 8px; }}
header p {{ opacity: 0.9; font-size: 14px; }}
.subtitle {{ font-size: 16px; opacity: 0.85; margin-top: 5px; }}

.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 30px; }}
.kpi {{ background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; }}
.kpi .value {{ font-size: 24px; font-weight: 700; color: #1a237e; white-space: nowrap; }}
.kpi .label {{ font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 1px; margin-top: 5px; }}
.kpi.risk .value {{ color: #d32f2f; }}
.kpi.safe .value {{ color: #2e7d32; }}

section {{ background: white; border-radius: 10px; padding: 30px; margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
section h2 {{ color: #1a237e; font-size: 20px; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #e3f2fd; }}
section h3 {{ color: #0d47a1; font-size: 16px; margin: 20px 0 12px; }}

.chart {{ text-align: center; margin: 20px 0; }}
.chart img {{ max-width: 100%; height: auto; border-radius: 8px; }}

table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin: 15px 0; }}
th {{ background: #e8eaf6; color: #1a237e; padding: 10px 8px; text-align: left; font-weight: 600; position: sticky; top: 0; }}
td {{ padding: 8px; border-bottom: 1px solid #f0f0f0; }}
tr:hover {{ background: #f5f5f5; }}
.num {{ text-align: right; font-family: 'Courier New', monospace; }}

.pd-bar {{ display: inline-block; height: 16px; border-radius: 3px; min-width: 2px; vertical-align: middle; }}
.pd-low {{ background: #4caf50; }}
.pd-med {{ background: #ff9800; }}
.pd-high {{ background: #f44336; }}
.pd-crit {{ background: #b71c1c; }}

.badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }}
.badge-aaa {{ background: #e8f5e9; color: #1b5e20; }}
.badge-a {{ background: #f1f8e9; color: #33691e; }}
.badge-bbb {{ background: #fff8e1; color: #f57f17; }}
.badge-bb {{ background: #fff3e0; color: #e65100; }}
.badge-b {{ background: #fbe9e7; color: #bf360c; }}
.badge-ccc {{ background: #ffebee; color: #b71c1c; }}
.badge-d {{ background: #f44336; color: white; }}

.grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
@media (max-width: 768px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}

.metric-box {{ background: #f8f9fa; border-radius: 8px; padding: 16px; margin: 8px 0; }}
.metric-box .metric-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
.metric-box .metric-value {{ font-size: 22px; font-weight: 700; color: #1a237e; }}

.sensitivity-row {{ display: flex; align-items: center; padding: 8px 0; border-bottom: 1px solid #f0f0f0; }}
.sensitivity-row .scenario {{ flex: 1; font-weight: 500; }}
.sensitivity-row .result {{ text-align: right; font-family: monospace; }}

footer {{ text-align: center; padding: 30px; color: #999; font-size: 12px; }}
</style>
</head>
<body>
<div class="container">

<header>
    <h1>Modelo Estrutural de Merton</h1>
    <p class="subtitle">Probabilidade de Default para Empresas Listadas na B3</p>
    <p>Gerado em {timestamp} | Dados: CVM DFP 2023-2024 | Horizonte: 1 ano</p>
</header>

<!-- KPIs -->
<div class="kpi-grid">
    <div class="kpi">
        <div class="value">{n_total}</div>
        <div class="label">Empresas Analisadas</div>
    </div>
    <div class="kpi">
        <div class="value">{n_converged}/{len(merton_results) if not merton_results.empty else 0}</div>
        <div class="label">Convergência Solver</div>
    </div>
    <div class="kpi risk">
        <div class="value">{pd_mean:.2f}%</div>
        <div class="label">PD Média (Merton)</div>
    </div>
    <div class="kpi">
        <div class="value">{pd_median:.2f}%</div>
        <div class="label">PD Mediana</div>
    </div>
    <div class="kpi safe">
        <div class="value">{dd_mean:.2f}</div>
        <div class="label">Distance-to-Default Média</div>
    </div>
    <div class="kpi">
        <div class="value">{n_sectors}</div>
        <div class="label">Setores Analisados</div>
    </div>
    <div class="kpi safe">
        <div class="value">{n_ig}</div>
        <div class="label">Investment Grade</div>
    </div>
    <div class="kpi risk">
        <div class="value">{n_distress}</div>
        <div class="label">Em Distress (PD&gt;15%)</div>
    </div>
</div>
'''

    # Seção 1: Modelo de Merton - Teoria
    html += '''
<section>
    <h2>1. Modelo Estrutural de Merton - Fundamentação</h2>
    <div class="grid-2">
        <div>
            <h3>Framework Teórico</h3>
            <p>O modelo de Merton (1974) trata o equity da firma como uma <strong>call option europeia</strong>
            sobre o valor dos ativos, com strike igual ao valor da dívida.</p>
            <ul style="margin: 10px 0 10px 20px; font-size: 13px;">
                <li>Valor da firma: <code>dV = μVdt + σVdW</code> (GBM)</li>
                <li>Default quando: <code>V(T) &lt; D</code></li>
                <li>Equity: <code>E = V·Φ(d₁) - D·e⁻ʳᵀ·Φ(d₂)</code></li>
                <li>Relação vol: <code>σE = (V/E)·Φ(d₁)·σV</code></li>
            </ul>
        </div>
        <div>
            <h3>Implementação</h3>
            <ul style="margin: 10px 0 10px 20px; font-size: 13px;">
                <li><strong>Solver:</strong> Iteração de ponto fixo (200 iter, tol=10⁻⁸)</li>
                <li><strong>Barreira:</strong> D = CP + 0.5 × LP (KMV)</li>
'''
    html += f'''                <li><strong>Taxa livre risco:</strong> SELIC = {SELIC*100:.2f}%</li>
                <li><strong>Drift:</strong> Risk-neutral (μ = r)</li>
                <li><strong>Horizonte:</strong> T = 1 ano</li>
                <li><strong>DD:</strong> <code>[ln(V/D) + (μ - 0.5σ²)T] / (σ√T)</code></li>
                <li><strong>PD:</strong> <code>Φ(-DD)</code></li>
            </ul>
        </div>
    </div>
</section>
'''

    # Seção 2: Convergência
    html += '<section>\n<h2>2. Convergência do Solver Numérico</h2>\n'
    if charts.get('convergence'):
        html += f'<div class="chart"><img src="data:image/png;base64,{charts["convergence"]}" alt="Convergência"></div>\n'

    conv_rate = n_converged / len(merton_results) * 100 if len(merton_results) > 0 else 0
    med_iter = merton_results[merton_results['converged']]['iterations'].median() if n_converged > 0 else 0
    html += f'''
    <div class="grid-2">
        <div class="metric-box">
            <div class="metric-label">Taxa de Convergência</div>
            <div class="metric-value">{conv_rate:.1f}%</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Mediana de Iterações</div>
            <div class="metric-value">{med_iter:.0f}</div>
        </div>
    </div>
</section>
'''

    # Seção 3: Distribuição PD e DD
    html += '<section>\n<h2>3. Distribuição da PD e Distance-to-Default</h2>\n<div class="grid-2">\n'
    if charts.get('pd_dist'):
        html += f'<div class="chart"><img src="data:image/png;base64,{charts["pd_dist"]}" alt="Distribuição PD"></div>\n'
    if charts.get('dd_dist'):
        html += f'<div class="chart"><img src="data:image/png;base64,{charts["dd_dist"]}" alt="Distribuição DD"></div>\n'
    html += '</div>\n'

    if charts.get('scatter'):
        html += f'<div class="chart"><img src="data:image/png;base64,{charts["scatter"]}" alt="Scatter"></div>\n'

    html += '</section>\n'

    # Seção 4: Rating Distribution
    html += '<section>\n<h2>4. Distribuição por Rating (Merton)</h2>\n'
    if charts.get('rating'):
        html += f'<div class="chart"><img src="data:image/png;base64,{charts["rating"]}" alt="Ratings"></div>\n'

    if not rating_dist.empty:
        html += '<table>\n<tr><th>Rating</th><th class="num">N</th><th class="num">%</th><th class="num">PD Média</th><th class="num">PD Mediana</th></tr>\n'
        for _, row in rating_dist.iterrows():
            badge_class = _rating_badge(row['rating'])
            html += f'<tr><td><span class="badge {badge_class}">{row["rating"]}</span></td>'
            html += f'<td class="num">{int(row["n"])}</td>'
            html += f'<td class="num">{row["pct"]:.1f}%</td>'
            html += f'<td class="num">{row["pd_media"]*100:.2f}%</td>'
            html += f'<td class="num">{row["pd_mediana"]*100:.2f}%</td></tr>\n'
        html += '</table>\n'
    html += '</section>\n'

    # Seção 5: PD por Setor
    html += '<section>\n<h2>5. PD Merton por Setor Econômico</h2>\n'
    if charts.get('pd_sector'):
        html += f'<div class="chart"><img src="data:image/png;base64,{charts["pd_sector"]}" alt="PD por Setor"></div>\n'

    if not sector_pd.empty:
        html += '<table>\n<tr><th>Setor</th><th class="num">N</th><th class="num">PD Média</th><th class="num">PD Mediana</th><th class="num">PD P90</th><th class="num">PD Pond.</th><th class="num">% Inv.Grade</th><th class="num">% Distress</th><th>Barra</th></tr>\n'
        for _, row in sector_pd.head(25).iterrows():
            pd_val = row['pd_media']
            bar_w = min(pd_val * 300, 150)
            bar_class = _pd_bar_class(pd_val)
            sector_name = str(row['sector'])[:35]
            html += f'<tr><td>{sector_name}</td>'
            html += f'<td class="num">{int(row["n_empresas"])}</td>'
            html += f'<td class="num">{pd_val*100:.2f}%</td>'
            html += f'<td class="num">{row["pd_mediana"]*100:.2f}%</td>'
            html += f'<td class="num">{row["pd_p90"]*100:.2f}%</td>'
            html += f'<td class="num">{row["pd_ponderada"]*100:.2f}%</td>'
            html += f'<td class="num">{row["pct_investment_grade"]*100:.0f}%</td>'
            html += f'<td class="num">{row["pct_distress"]*100:.0f}%</td>'
            html += f'<td><span class="pd-bar {bar_class}" style="width:{bar_w}px"></span></td></tr>\n'
        html += '</table>\n'
    html += '</section>\n'

    # Seção 6: Heatmap
    html += '<section>\n<h2>6. Heatmap Intersetorial</h2>\n'
    if charts.get('heatmap'):
        html += f'<div class="chart"><img src="data:image/png;base64,{charts["heatmap"]}" alt="Heatmap"></div>\n'
    html += '</section>\n'

    # Seção 7: SHI
    html += '<section>\n<h2>7. Sectoral Health Index (SHI)</h2>\n'
    html += '''<p>O SHI é um indicador composto que combina Z-scores padronizados de rentabilidade (30%),
    alavancagem (25%), liquidez (25%) e volatilidade (20%). Escala: [-3, +3].</p>\n'''
    if charts.get('shi'):
        html += f'<div class="chart"><img src="data:image/png;base64,{charts["shi"]}" alt="SHI"></div>\n'

    if not shi_results.empty:
        html += '<table>\n<tr><th>Setor</th><th class="num">N</th><th class="num">Z Rent.</th><th class="num">Z Alav.</th><th class="num">Z Liq.</th><th class="num">Z Vol.</th><th class="num">SHI</th><th>Classificação</th></tr>\n'
        shi_sorted = shi_results.sort_values('shi_normalized')
        for _, row in shi_sorted.iterrows():
            color = row.get('color', '#999')
            html += f'<tr><td>{str(row["sector"])[:30]}</td>'
            html += f'<td class="num">{int(row["n_companies"])}</td>'
            html += f'<td class="num">{row["z_rentabilidade"]:+.2f}</td>'
            html += f'<td class="num">{row["z_alavancagem"]:+.2f}</td>'
            html += f'<td class="num">{row["z_liquidez"]:+.2f}</td>'
            html += f'<td class="num">{row["z_volatilidade"]:+.2f}</td>'
            html += f'<td class="num" style="color:{color};font-weight:700">{row["shi_normalized"]:+.2f}</td>'
            html += f'<td style="color:{color}">{row["classification"]}</td></tr>\n'
        html += '</table>\n'
    html += '</section>\n'

    # Seção 8: Análise Temporal
    if temporal_df is not None and not temporal_df.empty:
        html += _build_temporal_section(temporal_df, charts)

    # Seção 9: Top Distress
    html += '<section>\n<h2>9. Top 25 - Maior Risco de Default</h2>\n'
    if not top_distress.empty:
        html += '<table>\n<tr><th>#</th><th>Empresa</th><th>Setor</th><th class="num">PD Merton</th><th class="num">DD</th><th class="num">Altman Z</th><th class="num">D/E</th><th class="num">ROA</th><th>Rating</th></tr>\n'
        for i, (_, row) in enumerate(top_distress.iterrows(), 1):
            pd_val = row.get('pd_merton', 0)
            badge = _rating_badge(row.get('rating_bucket', 'D'))
            html += f'<tr><td>{i}</td>'
            html += f'<td>{str(row.get("DENOM_CIA", ""))[:30]}</td>'
            html += f'<td>{str(row.get("sector", ""))[:20]}</td>'
            html += f'<td class="num" style="color:#d32f2f;font-weight:700">{pd_val*100:.2f}%</td>'
            dd_val = row.get('distance_to_default', float('nan'))
            html += f'<td class="num">{dd_val:.2f}</td>' if not pd.isna(dd_val) else '<td class="num">-</td>'
            az = row.get('altman_zscore', float('nan'))
            html += f'<td class="num">{az:.2f}</td>' if not pd.isna(az) else '<td class="num">-</td>'
            de = row.get('debt_to_equity', float('nan'))
            html += f'<td class="num">{de:.2f}</td>' if not pd.isna(de) else '<td class="num">-</td>'
            roa_v = row.get('roa', float('nan'))
            html += f'<td class="num">{roa_v*100:.1f}%</td>' if not pd.isna(roa_v) else '<td class="num">-</td>'
            html += f'<td><span class="badge {badge}">{row.get("rating_bucket", "-")}</span></td></tr>\n'
        html += '</table>\n'
    html += '</section>\n'

    # Seção 9: Top Safe
    html += '<section>\n<h2>10. Top 20 - Empresas Mais Saudáveis</h2>\n'
    if not top_safe.empty:
        html += '<table>\n<tr><th>#</th><th>Empresa</th><th>Setor</th><th class="num">PD Merton</th><th class="num">DD</th><th class="num">Altman Z</th><th class="num">Liq. Corrente</th><th class="num">ROA</th><th>Rating</th></tr>\n'
        for i, (_, row) in enumerate(top_safe.iterrows(), 1):
            pd_val = row.get('pd_merton', 0)
            badge = _rating_badge(row.get('rating_bucket', ''))
            html += f'<tr><td>{i}</td>'
            html += f'<td>{str(row.get("DENOM_CIA", ""))[:30]}</td>'
            html += f'<td>{str(row.get("sector", ""))[:20]}</td>'
            html += f'<td class="num" style="color:#2e7d32;font-weight:700">{pd_val*100:.4f}%</td>'
            dd_val = row.get('distance_to_default', float('nan'))
            html += f'<td class="num">{dd_val:.2f}</td>' if not pd.isna(dd_val) else '<td class="num">-</td>'
            az = row.get('altman_zscore', float('nan'))
            html += f'<td class="num">{az:.2f}</td>' if not pd.isna(az) else '<td class="num">-</td>'
            cr = row.get('current_ratio', float('nan'))
            html += f'<td class="num">{cr:.2f}</td>' if not pd.isna(cr) else '<td class="num">-</td>'
            roa_v = row.get('roa', float('nan'))
            html += f'<td class="num">{roa_v*100:.1f}%</td>' if not pd.isna(roa_v) else '<td class="num">-</td>'
            html += f'<td><span class="badge {badge}">{row.get("rating_bucket", "-")}</span></td></tr>\n'
        html += '</table>\n'
    html += '</section>\n'

    # Seção 10: Validação
    html += '<section>\n<h2>11. Validação do Modelo</h2>\n'
    html += '<div class="grid-2">\n'

    # Backtest
    html += '<div>\n<h3>Poder Discriminante</h3>\n'
    html += '<table>\n'
    html += f'<tr><td>Estatística KS</td><td class="num"><strong>{backtest.ks_statistic:.4f}</strong></td></tr>\n'
    html += f'<tr><td>Coeficiente de Gini</td><td class="num"><strong>{backtest.gini_coefficient:.4f}</strong></td></tr>\n'
    html += f'<tr><td>Accuracy Ratio</td><td class="num"><strong>{backtest.accuracy_ratio:.4f}</strong></td></tr>\n'
    html += f'<tr><td>PD média (defaults)</td><td class="num">{backtest.avg_pd_defaults*100:.2f}%</td></tr>\n'
    html += f'<tr><td>PD média (non-defaults)</td><td class="num">{backtest.avg_pd_non_defaults*100:.2f}%</td></tr>\n'
    html += f'<tr><td>PD ratio</td><td class="num">{backtest.pd_ratio:.2f}x</td></tr>\n'
    html += f'<tr><td>N defaults (proxy)</td><td class="num">{backtest.n_defaults}</td></tr>\n'
    html += f'<tr><td>Taxa de default</td><td class="num">{backtest.default_rate*100:.2f}%</td></tr>\n'
    html += '</table>\n</div>\n'

    # Correlação
    html += '<div>\n<h3>Correlação PD vs Default</h3>\n'
    html += '<table>\n'
    html += f'<tr><td>Correlação de Spearman</td><td class="num"><strong>{correlation["rank_correlation"]:.4f}</strong></td></tr>\n'
    html += f'<tr><td>N eventos RJ/default</td><td class="num">{correlation["n_rj"]}</td></tr>\n'
    rc_rj = correlation["avg_pd_rj"]
    rc_nrj = correlation["avg_pd_non_rj"]
    html += f'<tr><td>PD média (RJ)</td><td class="num">{rc_rj*100:.2f}%</td></tr>\n' if not pd.isna(rc_rj) else ''
    html += f'<tr><td>PD média (non-RJ)</td><td class="num">{rc_nrj*100:.2f}%</td></tr>\n' if not pd.isna(rc_nrj) else ''
    html += '</table>\n</div>\n'
    html += '</div>\n'

    # Análise por Decil
    if not backtest.decile_analysis.empty:
        html += '<h3>Análise por Decil de PD</h3>\n'
        html += '<table>\n<tr><th>Decil</th><th class="num">N</th><th class="num">PD Média</th><th class="num">N Defaults</th><th class="num">Taxa Default</th></tr>\n'
        for _, row in backtest.decile_analysis.iterrows():
            html += f'<tr><td>{int(row["decile"])}</td>'
            html += f'<td class="num">{int(row["n"])}</td>'
            html += f'<td class="num">{row["pd_media"]*100:.2f}%</td>'
            html += f'<td class="num">{int(row["n_defaults"])}</td>'
            html += f'<td class="num">{row["default_rate_decile"]*100:.1f}%</td></tr>\n'
        html += '</table>\n'

    html += '</section>\n'

    # Seção 11: Sensibilidade
    html += '<section>\n<h2>12. Análise de Sensibilidade</h2>\n'
    html += '<p>Impacto de cenários macroeconômicos na PD média do portfólio.</p>\n'
    if not sensitivity.empty:
        html += '<table>\n<tr><th>Cenário</th><th class="num">Fator</th><th class="num">PD Média</th><th class="num">PD P90</th><th class="num">N Distress</th><th class="num">% Distress</th><th class="num">N Default</th></tr>\n'
        for _, row in sensitivity.iterrows():
            is_base = row['cenario'] == 'Base'
            style = ' style="font-weight:700;background:#e8eaf6"' if is_base else ''
            html += f'<tr{style}>'
            html += f'<td>{row["cenario"]}</td>'
            html += f'<td class="num">{row["fator"]:.2f}x</td>'
            html += f'<td class="num">{row["pd_media"]*100:.2f}%</td>'
            html += f'<td class="num">{row["pd_p90"]*100:.2f}%</td>'
            html += f'<td class="num">{int(row["n_distress"])}</td>'
            html += f'<td class="num">{row["pct_distress"]*100:.1f}%</td>'
            html += f'<td class="num">{int(row["n_default"])}</td></tr>\n'
        html += '</table>\n'
    html += '</section>\n'

    # Seção 12: Concentração
    html += '<section>\n<h2>13. Análise de Concentração</h2>\n'
    html += '<div class="grid-2">\n'
    html += '<div class="metric-box">\n'
    html += f'<div class="metric-label">HHI (Herfindahl-Hirschman)</div>\n'
    html += f'<div class="metric-value">{concentration.get("hhi", 0):.4f}</div>\n'
    html += f'<div style="font-size:12px;color:#666">{concentration.get("hhi_classification", "")}</div>\n'
    html += '</div>\n'
    html += '<div class="metric-box">\n'
    html += f'<div class="metric-label">Setores Analisados</div>\n'
    html += f'<div class="metric-value">{concentration.get("n_sectors", 0)}</div>\n'
    html += '</div>\n</div>\n'

    # Top risco
    top_risk = concentration.get('top_risk_sectors', [])
    if top_risk:
        html += '<h3>Setores de Maior Risco</h3>\n<table>\n<tr><th>Setor</th><th class="num">PD Média</th><th class="num">N</th></tr>\n'
        for s in top_risk:
            html += f'<tr><td>{s["sector"][:35]}</td>'
            html += f'<td class="num" style="color:#d32f2f">{s["pd_media"]*100:.2f}%</td>'
            html += f'<td class="num">{int(s["n_empresas"])}</td></tr>\n'
        html += '</table>\n'

    top_safe_sectors = concentration.get('top_safe_sectors', [])
    if top_safe_sectors:
        html += '<h3>Setores Mais Saudáveis</h3>\n<table>\n<tr><th>Setor</th><th class="num">PD Média</th><th class="num">N</th></tr>\n'
        for s in top_safe_sectors:
            html += f'<tr><td>{s["sector"][:35]}</td>'
            html += f'<td class="num" style="color:#2e7d32">{s["pd_media"]*100:.2f}%</td>'
            html += f'<td class="num">{int(s["n_empresas"])}</td></tr>\n'
        html += '</table>\n'

    html += '</section>\n'

    # Seção 13: Estatísticas Descritivas
    html += '<section>\n<h2>14. Estatísticas Descritivas</h2>\n'
    desc_cols = {
        'pd_merton': 'PD Merton',
        'distance_to_default': 'Distance-to-Default',
        'asset_volatility': 'Vol. Ativos',
        'leverage_ratio': 'Leverage (D/V)',
        'altman_zscore': 'Altman Z',
        'current_ratio': 'Liq. Corrente',
        'debt_to_equity': 'D/E',
        'roa': 'ROA',
        'ebit_margin': 'Margem EBIT',
    }
    available_desc = {k: v for k, v in desc_cols.items() if k in dataset.columns}

    if available_desc:
        desc_df = dataset[list(available_desc.keys())].describe(
            percentiles=[0.05, 0.25, 0.5, 0.75, 0.90, 0.95]
        )
        html += '<table>\n<tr><th>Estatística</th>'
        for col_name in available_desc.values():
            html += f'<th class="num">{col_name}</th>'
        html += '</tr>\n'
        stat_labels = {
            'count': 'N', 'mean': 'Média', 'std': 'Desvio',
            'min': 'Min', '5%': 'P5', '25%': 'P25',
            '50%': 'Mediana', '75%': 'P75', '90%': 'P90',
            '95%': 'P95', 'max': 'Max',
        }
        for stat in desc_df.index:
            label = stat_labels.get(stat, stat)
            html += f'<tr><td>{label}</td>'
            for col in available_desc.keys():
                val = desc_df.loc[stat, col]
                if stat == 'count':
                    html += f'<td class="num">{int(val)}</td>'
                elif col in ('pd_merton',):
                    html += f'<td class="num">{val*100:.3f}%</td>'
                elif col in ('roa', 'ebit_margin'):
                    html += f'<td class="num">{val*100:.2f}%</td>'
                else:
                    html += f'<td class="num">{val:.3f}</td>'
            html += '</tr>\n'
        html += '</table>\n'
    html += '</section>\n'

    # Seção 14: Notas Metodológicas
    html += '''
<section>
    <h2>15. Notas Metodológicas</h2>
    <div class="grid-2">
        <div>
            <h3>Modelo de Merton</h3>
            <ul style="margin: 10px 0 10px 20px; font-size: 13px;">
                <li><strong>Base teórica:</strong> Merton (1974), "On the Pricing of Corporate Debt"</li>
                <li><strong>Equação fundamental:</strong> E = V·Φ(d₁) - D·e⁻ʳᵀ·Φ(d₂)</li>
                <li><strong>Solver:</strong> Iteração de ponto fixo com convergência quadrática</li>
                <li><strong>Barreira de default:</strong> D = Dívida CP + 0.5 × Dívida LP (convenção KMV/Moody's)</li>
                <li><strong>Equity value:</strong> Patrimônio Líquido contábil como proxy (na ausência de market cap)</li>
                <li><strong>Volatilidade do equity:</strong> Estimada por setor via heurísticas de mercado</li>
                <li><strong>Setor financeiro:</strong> Barreira de dívida ajustada com fator de {FINANCIAL_DEBT_BARRIER_FACTOR:.0%}
                    + cap adaptativo de D/E≤{MAX_FINANCIAL_DE:.0f}× para bancos, seguradoras, corretoras e holdings,
                    refletindo que depósitos, captações e passivos regulatórios não representam dívida "distress" no sentido estrutural</li>
                <li><strong>Limitações:</strong> Ausência de dados de mercado (preço/volume) para calibração mais precisa</li>
            </ul>
        </div>
        <div>
            <h3>Indicadores Complementares</h3>
            <ul style="margin: 10px 0 10px 20px; font-size: 13px;">
                <li><strong>Altman Z-Score:</strong> Versão para mercados emergentes (1995)</li>
                <li><strong>SHI:</strong> Índice composto setorial com 4 dimensões padronizadas</li>
                <li><strong>Fonte de dados:</strong> CVM - Portal de Dados Abertos (DFP 2023-2024)</li>
                <li><strong>Tratamento de escala:</strong> Conversão MIL→×1000, UNIDADE→×1</li>
                <li><strong>Deduplicação:</strong> Versão mais recente preservada por empresa/conta</li>
                <li><strong>Validação:</strong> KS, Gini, análise por decil, sensibilidade a choques</li>
                <li><strong>Viés de sobrevivência:</strong> Inclusão de empresas canceladas no cadastro CVM</li>
            </ul>
        </div>
    </div>
</section>
'''

    html += f'''
<footer>
    <p>Relatório gerado automaticamente em {timestamp}</p>
    <p>Modelo Estrutural de Merton | PD_B3 Analytics Engine</p>
    <p>Dados: CVM (dados.cvm.gov.br) | Python + NumPy + SciPy</p>
</footer>

</div>
</body>
</html>'''

    return html


# ============================================================
# HELPERS
# ============================================================

def _build_temporal_section(temporal_df: pd.DataFrame, charts: dict) -> str:
    """Constrói seção HTML da análise temporal."""
    html = '<section>\n<h2>8. Análise Temporal - Evolução Setorial (2023 → 2024)</h2>\n'
    html += '<p>Comparação da PD Merton e indicadores fundamentalistas entre os exercícios de 2023 e 2024, '
    html += 'identificando setores em melhoria e deterioração.</p>\n'

    # KPIs temporais
    n_improved = (temporal_df['delta_pd'] < -0.005).sum()
    n_stable = ((temporal_df['delta_pd'] >= -0.005) & (temporal_df['delta_pd'] <= 0.005)).sum()
    n_worsened = (temporal_df['delta_pd'] > 0.005).sum()
    avg_delta = temporal_df['delta_pd'].mean() * 100
    max_improve = temporal_df['delta_pd'].min() * 100
    max_worsen = temporal_df['delta_pd'].max() * 100

    html += '<div class="kpi-grid">\n'
    html += f'<div class="kpi safe"><div class="value">{n_improved}</div><div class="label">Setores em Melhoria</div></div>\n'
    html += f'<div class="kpi"><div class="value">{n_stable}</div><div class="label">Setores Estáveis</div></div>\n'
    html += f'<div class="kpi risk"><div class="value">{n_worsened}</div><div class="label">Setores em Deterioração</div></div>\n'
    html += f'<div class="kpi"><div class="value">{avg_delta:+.2f}pp</div><div class="label">Δ PD Médio</div></div>\n'
    html += '</div>\n'

    # Gráfico de variação PD
    if charts.get('temporal_pd'):
        html += '<h3>Variação da PD por Setor</h3>\n'
        html += f'<div class="chart"><img src="data:image/png;base64,{charts["temporal_pd"]}" alt="Variação PD"></div>\n'

    # Gráfico de migração
    if charts.get('migration'):
        html += '<h3>Migração de PD (Scatter 2023 vs 2024)</h3>\n'
        html += f'<div class="chart"><img src="data:image/png;base64,{charts["migration"]}" alt="Migração"></div>\n'

    # Gráfico de indicadores
    if charts.get('temporal_indicators'):
        html += '<h3>Variação de Indicadores Fundamentalistas</h3>\n'
        html += f'<div class="chart"><img src="data:image/png;base64,{charts["temporal_indicators"]}" alt="Indicadores"></div>\n'

    # Tabela detalhada
    html += '<h3>Detalhamento por Setor</h3>\n'
    html += '<table>\n<tr><th>Setor</th><th class="num">N</th>'
    html += '<th class="num">PD 2023</th><th class="num">PD 2024</th>'
    html += '<th class="num">Δ PD</th><th class="num">Δ DD</th>'
    html += '<th>Tendência</th></tr>\n'

    for _, row in temporal_df.iterrows():
        delta = row['delta_pd']
        color = '#4caf50' if delta < -0.005 else ('#f44336' if delta > 0.005 else '#9e9e9e')
        arrow = '↓' if delta < -0.005 else ('↑' if delta > 0.005 else '→')
        trend = row.get('trend', '')

        html += f'<tr><td>{str(row["sector"])[:30]}</td>'
        html += f'<td class="num">{int(row["n_common"])}</td>'
        html += f'<td class="num">{row["pd_2023"]*100:.2f}%</td>'
        html += f'<td class="num">{row["pd_2024"]*100:.2f}%</td>'
        html += f'<td class="num" style="color:{color};font-weight:700">{arrow} {delta*100:+.2f}pp</td>'
        delta_dd = row.get('delta_dd', float('nan'))
        if pd.notna(delta_dd):
            dd_color = '#4caf50' if delta_dd > 0 else '#f44336'
            html += f'<td class="num" style="color:{dd_color}">{delta_dd:+.2f}</td>'
        else:
            html += '<td class="num">-</td>'
        html += f'<td style="color:{color}">{trend}</td></tr>\n'

    html += '</table>\n'

    # Resumo top movers
    top_improve = temporal_df.nsmallest(5, 'delta_pd')
    top_worsen = temporal_df.nlargest(5, 'delta_pd')

    html += '<div class="grid-2">\n'
    html += '<div>\n<h3 style="color:#2e7d32">Top 5 - Maior Melhoria</h3>\n<table>\n'
    html += '<tr><th>Setor</th><th class="num">Δ PD</th></tr>\n'
    for _, row in top_improve.iterrows():
        html += f'<tr><td>{str(row["sector"])[:30]}</td>'
        html += f'<td class="num" style="color:#2e7d32;font-weight:700">{row["delta_pd"]*100:+.2f}pp</td></tr>\n'
    html += '</table>\n</div>\n'

    html += '<div>\n<h3 style="color:#d32f2f">Top 5 - Maior Deterioração</h3>\n<table>\n'
    html += '<tr><th>Setor</th><th class="num">Δ PD</th></tr>\n'
    for _, row in top_worsen.iterrows():
        html += f'<tr><td>{str(row["sector"])[:30]}</td>'
        html += f'<td class="num" style="color:#d32f2f;font-weight:700">{row["delta_pd"]*100:+.2f}pp</td></tr>\n'
    html += '</table>\n</div>\n</div>\n'

    html += '</section>\n'
    return html


def _rating_badge(rating: str) -> str:
    r = str(rating).upper()
    if r in ('AAA', 'AA'):
        return 'badge-aaa'
    elif r == 'A':
        return 'badge-a'
    elif r == 'BBB':
        return 'badge-bbb'
    elif r == 'BB':
        return 'badge-bb'
    elif r == 'B':
        return 'badge-b'
    elif r in ('CCC', 'CC'):
        return 'badge-ccc'
    return 'badge-d'


def _pd_bar_class(pd_val: float) -> str:
    if pd_val < 0.05:
        return 'pd-low'
    elif pd_val < 0.15:
        return 'pd-med'
    elif pd_val < 0.35:
        return 'pd-high'
    return 'pd-crit'


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    run_pipeline()
