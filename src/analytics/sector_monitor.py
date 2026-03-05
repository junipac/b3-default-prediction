"""
Monitoramento de Desempenho Setorial.

Módulo para construção de índices setoriais sintéticos, indicadores
de performance rolling, e métricas de deterioração/melhoria.

Funcionalidades:
    - Índice setorial base 100 (equal-weighted e value-weighted)
    - Retorno acumulado, volatilidade rolling, drawdown máximo
    - EBITDA médio ponderado, alavancagem, cobertura de juros
    - Crescimento de receita, distância ao default média
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class WeightingMethod(Enum):
    EQUAL = "equal_weighted"
    VALUE = "value_weighted"


@dataclass
class SectorMetrics:
    """Métricas consolidadas de um setor em um período."""
    sector: str
    period: str
    n_companies: int
    cumulative_return: float
    rolling_volatility_12m: float
    max_drawdown: float
    avg_ebitda_weighted: float
    avg_leverage: float
    avg_interest_coverage: float
    avg_revenue_growth: float
    avg_distance_to_default: float
    index_value: float


class SectorMonitor:
    """
    Monitor de desempenho setorial.

    Constrói índices sintéticos e métricas consolidadas por setor.
    """

    def __init__(self, weighting: WeightingMethod = WeightingMethod.EQUAL):
        self.weighting = weighting

    def build_sector_index(
        self,
        df: pd.DataFrame,
        base_value: float = 100.0,
    ) -> pd.DataFrame:
        """
        Constrói índice setorial sintético base 100.

        Args:
            df: DataFrame com colunas: sector, reference_date, equity, total_assets,
                net_revenue, ebit, net_income, plus indicadores derivados.
            base_value: Valor base do índice.

        Returns:
            DataFrame com série temporal do índice por setor.
        """
        if df.empty:
            return pd.DataFrame()

        # Normalizar colunas
        df = df.copy()
        sector_col = self._find_column(df, ['sector', 'SETOR_ATIV'])
        date_col = self._find_column(df, ['reference_date', 'DT_REFER'])

        if not sector_col or not date_col:
            logger.warning("Colunas de setor/data não encontradas")
            return pd.DataFrame()

        df['_sector'] = df[sector_col].fillna('Não Classificado')
        df['_date'] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=['_date'])

        # Métrica base para o índice: usar ROA como proxy de performance
        roa_col = self._find_column(df, ['roa', 'ROA'])
        equity_col = self._find_column(df, ['equity', 'EQUITY'])
        assets_col = self._find_column(df, ['total_assets', 'TOTAL_ASSETS'])

        results = []

        for sector, group in df.groupby('_sector'):
            group = group.sort_values('_date')
            periods = group['_date'].unique()

            index_val = base_value
            for i, period in enumerate(sorted(periods)):
                period_data = group[group['_date'] == period]

                if self.weighting == WeightingMethod.VALUE and assets_col:
                    weights = period_data[assets_col].fillna(0)
                    total_w = weights.sum()
                    if total_w > 0:
                        weights = weights / total_w
                    else:
                        weights = pd.Series(
                            1.0 / len(period_data), index=period_data.index
                        )
                else:
                    weights = pd.Series(
                        1.0 / len(period_data), index=period_data.index
                    )

                # Retorno do período (baseado em ROA)
                if roa_col and roa_col in period_data.columns:
                    roa_vals = period_data[roa_col].fillna(0).clip(-1, 1)
                    period_return = (roa_vals * weights).sum()
                else:
                    period_return = 0.0

                if i > 0:
                    index_val = index_val * (1 + period_return)

                results.append({
                    'sector': sector,
                    'period': period,
                    'index_value': index_val,
                    'period_return': period_return,
                    'n_companies': len(period_data),
                })

        return pd.DataFrame(results)

    def compute_sector_metrics(
        self,
        df: pd.DataFrame,
        dd_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Calcula métricas consolidadas por setor.

        Args:
            df: DataFrame analítico com indicadores financeiros.
            dd_df: DataFrame com distance-to-default (Merton).

        Returns:
            DataFrame com métricas por setor.
        """
        if df.empty:
            return pd.DataFrame()

        df = df.copy()
        sector_col = self._find_column(df, ['sector', 'SETOR_ATIV'])
        if not sector_col:
            return pd.DataFrame()

        df['_sector'] = df[sector_col].fillna('Não Classificado')

        # Mapear colunas
        col_map = {
            'ebitda': self._find_column(df, ['ebitda_approx', 'EBITDA']),
            'leverage': self._find_column(df, ['debt_to_equity', 'DEBT_TO_EQUITY']),
            'coverage': self._find_column(df, ['interest_coverage', 'debt_coverage']),
            'revenue': self._find_column(df, ['net_revenue', 'NET_REVENUE']),
            'roa': self._find_column(df, ['roa', 'ROA']),
            'assets': self._find_column(df, ['total_assets', 'TOTAL_ASSETS']),
            'equity': self._find_column(df, ['equity', 'EQUITY']),
            'current_ratio': self._find_column(df, ['current_ratio', 'CURRENT_RATIO']),
            'ebit_margin': self._find_column(df, ['ebit_margin', 'EBIT_MARGIN']),
            'altman': self._find_column(df, ['altman_zscore', 'ALTMAN_ZSCORE']),
        }

        # Merge DD se disponível
        if dd_df is not None and not dd_df.empty:
            cnpj_col = self._find_column(df, ['cnpj_cia', 'CNPJ_CIA', 'cnpj'])
            if cnpj_col and 'cnpj' in dd_df.columns:
                dd_merge = dd_df[['cnpj', 'distance_to_default', 'pd_merton']].copy()
                dd_merge = dd_merge.rename(columns={'cnpj': cnpj_col})
                df = df.merge(dd_merge, on=cnpj_col, how='left')

        results = []
        for sector, group in df.groupby('_sector'):
            n = len(group)
            metrics = {'sector': sector, 'n_companies': n}

            # EBITDA médio ponderado
            if col_map['ebitda'] and col_map['assets']:
                ebitda = group[col_map['ebitda']].fillna(0)
                assets = group[col_map['assets']].fillna(0)
                total_assets = assets.sum()
                if total_assets > 0:
                    metrics['ebitda_medio_ponderado'] = (
                        (ebitda * assets).sum() / total_assets
                    )
                else:
                    metrics['ebitda_medio_ponderado'] = ebitda.mean()
            else:
                metrics['ebitda_medio_ponderado'] = float('nan')

            # Alavancagem média
            if col_map['leverage']:
                lev = group[col_map['leverage']].dropna()
                lev = lev[lev.between(-10, 50)]  # clip outliers
                metrics['alavancagem_media'] = lev.median() if len(lev) > 0 else float('nan')
            else:
                metrics['alavancagem_media'] = float('nan')

            # Cobertura de juros média
            if col_map['coverage']:
                cov = group[col_map['coverage']].dropna()
                cov = cov[cov.between(-20, 100)]
                metrics['cobertura_juros_media'] = cov.median() if len(cov) > 0 else float('nan')
            else:
                metrics['cobertura_juros_media'] = float('nan')

            # ROA médio
            if col_map['roa']:
                roa = group[col_map['roa']].dropna()
                metrics['roa_medio'] = roa.median() if len(roa) > 0 else float('nan')
            else:
                metrics['roa_medio'] = float('nan')

            # Margem EBIT média
            if col_map['ebit_margin']:
                em = group[col_map['ebit_margin']].dropna()
                em = em[em.between(-5, 5)]
                metrics['ebit_margin_media'] = em.median() if len(em) > 0 else float('nan')
            else:
                metrics['ebit_margin_media'] = float('nan')

            # Liquidez corrente média
            if col_map['current_ratio']:
                cr = group[col_map['current_ratio']].dropna()
                cr = cr[cr.between(0, 20)]
                metrics['liquidez_corrente_media'] = cr.median() if len(cr) > 0 else float('nan')
            else:
                metrics['liquidez_corrente_media'] = float('nan')

            # Altman Z médio
            if col_map['altman']:
                az = group[col_map['altman']].dropna()
                az = az[az.between(-20, 30)]
                metrics['altman_z_medio'] = az.median() if len(az) > 0 else float('nan')
            else:
                metrics['altman_z_medio'] = float('nan')

            # Distance to Default média (Merton)
            if 'distance_to_default' in group.columns:
                dd = group['distance_to_default'].dropna()
                metrics['dd_media'] = dd.median() if len(dd) > 0 else float('nan')
            else:
                metrics['dd_media'] = float('nan')

            # PD Merton média
            if 'pd_merton' in group.columns:
                pdm = group['pd_merton'].dropna()
                metrics['pd_merton_media'] = pdm.mean() if len(pdm) > 0 else float('nan')
                metrics['pd_merton_mediana'] = pdm.median() if len(pdm) > 0 else float('nan')
                metrics['pd_merton_p90'] = pdm.quantile(0.9) if len(pdm) > 0 else float('nan')
            else:
                metrics['pd_merton_media'] = float('nan')
                metrics['pd_merton_mediana'] = float('nan')
                metrics['pd_merton_p90'] = float('nan')

            # Ativo total do setor
            if col_map['assets']:
                metrics['ativo_total_setor'] = group[col_map['assets']].fillna(0).sum()
            else:
                metrics['ativo_total_setor'] = 0

            # % PL negativo
            if col_map['equity']:
                eq = group[col_map['equity']].dropna()
                metrics['pct_pl_negativo'] = (
                    (eq < 0).sum() / len(eq) if len(eq) > 0 else 0
                )
            else:
                metrics['pct_pl_negativo'] = float('nan')

            results.append(metrics)

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values('n_companies', ascending=False)
        return result_df

    def compute_rolling_metrics(
        self,
        df: pd.DataFrame,
        window: int = 4,
    ) -> pd.DataFrame:
        """
        Calcula métricas rolling (volatilidade, drawdown) por setor.

        Para dados trimestrais, window=4 = 12 meses.
        """
        sector_col = self._find_column(df, ['sector', 'SETOR_ATIV'])
        date_col = self._find_column(df, ['reference_date', 'DT_REFER'])
        roa_col = self._find_column(df, ['roa', 'ROA'])

        if not all([sector_col, date_col, roa_col]):
            return pd.DataFrame()

        df = df.copy()
        df['_date'] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=['_date'])

        results = []
        for sector, group in df.groupby(sector_col):
            group = group.sort_values('_date')

            # ROA médio por período
            period_roa = group.groupby('_date')[roa_col].mean().reset_index()
            period_roa.columns = ['period', 'avg_roa']
            period_roa = period_roa.sort_values('period')

            if len(period_roa) < 2:
                continue

            # Índice acumulado
            period_roa['index'] = (1 + period_roa['avg_roa'].fillna(0)).cumprod() * 100

            # Volatilidade rolling
            period_roa['vol_rolling'] = (
                period_roa['avg_roa'].rolling(window=window, min_periods=2).std()
                * np.sqrt(4)  # anualizar (trimestral)
            )

            # Drawdown
            period_roa['peak'] = period_roa['index'].cummax()
            period_roa['drawdown'] = (
                (period_roa['index'] - period_roa['peak']) / period_roa['peak']
            )
            period_roa['max_drawdown'] = period_roa['drawdown'].cummin()

            # Retorno acumulado
            if period_roa['index'].iloc[0] > 0:
                period_roa['cum_return'] = (
                    period_roa['index'] / period_roa['index'].iloc[0] - 1
                )
            else:
                period_roa['cum_return'] = 0

            period_roa['sector'] = sector

            results.append(period_roa)

        if not results:
            return pd.DataFrame()

        return pd.concat(results, ignore_index=True)

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Encontra primeira coluna existente entre candidatos."""
        for c in candidates:
            if c in df.columns:
                return c
        # Busca case-insensitive
        lower_cols = {col.lower(): col for col in df.columns}
        for c in candidates:
            if c.lower() in lower_cols:
                return lower_cols[c.lower()]
        return None
